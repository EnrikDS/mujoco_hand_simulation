import mujoco

def bind_grid(model: mujoco.MjModel, name: str, nchannel: int, nx: int, ny: int):
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
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise RuntimeError(f"Sensor '{name}' not found")
    adr = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    need = nchannel * nx * ny
    if dim != need:
        raise RuntimeError(f"{name}: expected dim {need}, got {dim}")
    return sid, adr, dim, (nchannel, nx, ny)

class TimePlot:
    """Rolling time window plot with three curves: Fx, Fy, Fz."""
    def __init__(self, title: str, horizon_s: float, dt: float):
        import numpy as np
        self.N = max(200, int(horizon_s / max(1e-6, dt)))
        self.t  = np.full(self.N, np.nan)
        self.fx = np.full(self.N, np.nan)
        self.fy = np.full(self.N, np.nan)
        self.fz = np.full(self.N, np.nan)
        self.i = 0

        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        self.ax.set_xlabel("time [s]")
        self.ax.set_ylabel("force [N]")
        (self.lx,) = self.ax.plot([], [], label="Fx")
        (self.ly,) = self.ax.plot([], [], label="Fy")
        (self.lz,) = self.ax.plot([], [], label="Fz")
        self.ax.legend(loc="upper right")

    def append(self, t, fx, fy, fz):
        k = self.i % self.N
        self.t[k]  = t
        self.fx[k] = fx
        self.fy[k] = fy
        self.fz[k] = fz
        self.i += 1

    def refresh(self):
        if self.i < 2:
            return
        n = min(self.i, self.N)
        idx = (np.arange(n) + self.i - n) % self.N
        tt = self.t[idx]
        self.lx.set_data(tt, self.fx[idx])
        self.ly.set_data(tt, self.fy[idx])
        self.lz.set_data(tt, self.fz[idx])
        self.ax.relim(); self.ax.autoscale_view()
        self.fig.canvas.draw(); self.fig.canvas.flush_events()