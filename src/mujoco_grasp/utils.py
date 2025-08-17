import mujoco
import numpy as np
import matplotlib.pyplot as plt
import argparse



def site_pose(data: mujoco.MjData, model: mujoco.MjModel, site_name: str):
    """
    Return the **world-frame pose** (position and rotation matrix) of a MuJoCo site.

    This helper looks up `site_name` in the compiled model, then reads:
      - `data.site_xpos[sid]` → the site's 3D position in **world** coordinates,
      - `data.site_xmat[sid]` → the site's 3×3 **world** rotation matrix (row-major).

    Both arrays are **copied** so subsequent simulation steps won’t mutate what you keep.

    Parameters
    ----------
    data : mujoco.MjData
        Runtime state associated with `model`. Must be **up-to-date**; call
        `mujoco.mj_forward(model, data)` (or step the sim) before using this,
        otherwise positions/orientations may be stale.
    model : mujoco.MjModel
        Compiled model that contains the site definition.
    site_name : str
        The exact name of the site (as defined in the MJCF XML).

    Returns
    -------
    pos : numpy.ndarray, shape (3,)
        World position of the site (meters).
    R : numpy.ndarray, shape (3, 3)
        World rotation matrix of the site. Each column is the site’s X, Y, Z
        basis vector expressed in world coordinates (MuJoCo stores it row-major
        flattened; this function reshapes it to (3,3)).

        Tips:
        - To convert a site-local vector `v_site` to world: `v_world = pos + R @ v_site`.
        - To express a world vector in site coordinates: `v_site = R.T @ (v_world - pos)`.
        - To get a quaternion (w, x, y, z): 
          ```python
          quat = np.empty(4)
          mujoco.mju_mat2Quat(quat, R)
          ```

    Raises
    ------
    RuntimeError
        If `site_name` does not exist in `model`.

    Notes
    -----
    - If you call this frequently inside a loop, **cache the site id** once:
      `sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)`
      and read `data.site_xpos[sid]`, `data.site_xmat[sid]` directly.
    - For tactile sensors like `touch_grid`, ensure the site’s **−Z axis** points
      outward from the surface you want to sense (blue +Z arrow should point *into*
      the body when visualizing frames).

    Example
    -------
    >>> mujoco.mj_forward(model, data)          # ensure transforms are current
    >>> pos, R = site_pose(data, model, "palm_grid_site")
    >>> tip_world = pos + R @ np.array([0.0, 0.0, 0.02])  # 2 cm along site +Z
    """
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise RuntimeError(f"Site '{site_name}' not found")
    pos = data.site_xpos[sid].copy()
    R = data.site_xmat[sid].reshape(3, 3).copy()
    return pos, R

class GridSensorBinding:
    """Bind a touch_grid sensor by name and validate its shape."""
    def __init__(self, model, name: str, nchannel: int, nx: int, ny: int):
        self.name = name
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise RuntimeError(f"Sensor '{name}' not found")
        self.adr = model.sensor_adr[sid]
        self.dim = model.sensor_dim[sid]
        self.nchannel, self.nx, self.ny = nchannel, nx, ny
        need = nchannel * nx * ny
        if self.dim != need:
            raise RuntimeError(f"{name}: expected dim {need}, got {self.dim}")

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
        # roll buffer so time increases left→right
        n = min(self.i, self.N)
        idx = (np.arange(n) + self.i - n) % self.N
        tt = self.t[idx]
        self.lx.set_data(tt, self.fx[idx])
        self.ly.set_data(tt, self.fy[idx])
        self.lz.set_data(tt, self.fz[idx])
        self.ax.relim(); self.ax.autoscale_view()
        self.fig.canvas.draw(); self.fig.canvas.flush_events()


def mj_reset_to_qpos(model, data, qpos_full):
    """Hard reset to a full qpos vector; zero everything; forward."""
    if len(qpos_full) != model.nq:
        raise RuntimeError(f"qpos length {len(qpos_full)} != model.nq {model.nq}")
    data.qpos[:] = qpos_full
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

def place_free_body_at_pose(model, data, body_name, pos_w, R_w):
    """Place a free-jointed body at world pose (pos_w, R_w) and zero its velocities."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise RuntimeError(f"Body '{body_name}' not found")
    jadr = model.body_jntadr[bid]
    if model.jnt_type[jadr] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError(f"Body '{body_name}' does not have a free joint")
    qadr = model.jnt_qposadr[jadr]
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, R_w.flatten(order='C'))       # accepts 3x3 on 3.3.5
    data.qpos[qadr:qadr+3]     = pos_w
    data.qpos[qadr+3:qadr+7]   = quat
    data.qvel[qadr:qadr+6]     = 0.0
    mujoco.mj_forward(model, data)
