import mujoco

def bind_grid(model: mujoco.MjModel, name: str, sensor_shape ):
    """
    Look up a MuJoCo touch_grid sensor by name and return everything needed to
    read it efficiently from `data.sensordata`.

    What it does, step by step:
      1) Finds the sensor's internal ID by `name` (mj_name2id).
      2) Fetches the starting index (address) of that sensor's data in the
         flat `data.sensordata` array (model.sensor_adr[sid]).
      3) Fetches the total number of scalars the sensor writes (model.sensor_dim[sid]).
      4) Verifies that this dimension matches what you expect for a touch_grid:
         nchannel * nx * ny (channels × rows × cols). If not, it raises.
      5) Returns the sensor id, its address, its dimension, and a convenient
         shape tuple (nchannel, ny, nx) you can use to reshape slices.

    Parameters
    ----------
    model : mujoco.MjModel
        The compiled model loaded from your XML.
    name : str
        The sensor's name as defined in the XML (`<plugin name="...">`).
    nchannel : int
        Number of channels per cell (e.g., 1 for Fz only, or 3 for Fz/Fx/Fy).
    nx : int
        Number of columns in the grid.
    ny : int
        Number of rows in the grid.

    Returns
    -------
    sid : int
        Internal MuJoCo sensor id (for debugging/introspection).
    adr : int
        Start index into `data.sensordata` where this sensor's block begins.
    dim : int
        Total number of scalars for this sensor (should be nchannel*nx*ny).
    shape : tuple[int, int, int]
        A reshape-friendly tuple (nchannel, ny, nx).

    Raises
    ------
    RuntimeError
        If the sensor name doesn't exist in the model, or if the actual sensor
        dimension in the compiled model doesn't match `nchannel*nx*ny`.

    Usage example
    -------------
    # Bind once (after mj_forward):
    sid, adr, dim, shape = bind_grid(model, "palm_grid", 1, 12, 8)

    # Each simulation step:
    grid_flat = data.sensordata[adr : adr + dim]
    grid = grid_flat.reshape(shape)    # shape = (1, 12, 8)
    Fz = grid[0]                       # normal force map (12×8)
    """
    nchannel, ny, nx = sensor_shape
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise RuntimeError(f"Sensor '{name}' not found")
    adr = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    need = nchannel * nx * ny
    if dim != need:
        raise RuntimeError(f"{name}: expected dim {need}, got {dim}")
    return sid, adr, dim, (nchannel, nx, ny)

