import mujoco
import numpy as np
import matplotlib.pyplot as plt
import math


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





class ForceGridPlot:
    """
    One Figure with N subplots (one per sensor), each showing Fx, Fy, Fz vs time
    using a rolling window. Call append(t, values) each update where
    values is a dict: {label: (Fx, Fy, Fz)}.
    """
    def __init__(self, labels, horizon_s: float, dt: float):
        self.labels = list(labels)
        self.N = max(200, int(horizon_s / max(1e-6, dt)))
        self.i = 0

        # ring buffers per sensor
        self.t  = np.full(self.N, np.nan)
        self.fx = {k: np.full(self.N, np.nan) for k in self.labels}
        self.fy = {k: np.full(self.N, np.nan) for k in self.labels}
        self.fz = {k: np.full(self.N, np.nan) for k in self.labels}

        # layout: up to 3 columns; rows as needed
        n = len(self.labels)
        cols = min(3, n)
        rows = int(math.ceil(n / cols))

        self.fig, axes = plt.subplots(rows, cols, sharex=True, figsize=(4.5*cols, 3.2*rows))
        if isinstance(axes, np.ndarray):
            self.axes = axes.ravel()
        else:
            self.axes = [axes]

        # create lines per subplot
        self.lines = {}
        for ax, label in zip(self.axes, self.labels):
            ax.set_title(label.upper())
            ax.set_xlabel("time [s]")
            ax.set_ylabel("force [N]")
            (lx,) = ax.plot([], [], label="Fx")
            (ly,) = ax.plot([], [], label="Fy")
            (lz,) = ax.plot([], [], label="Fz")
            ax.legend(loc="upper right")
            self.lines[label] = (lx, ly, lz)

        # hide any unused axes
        for j in range(len(self.labels), len(self.axes)):
            self.axes[j].set_visible(False)

        plt.ion()
        plt.show(block=False)

    def append(self, t: float, values: dict[str, tuple[float, float, float]]):
        """values: {label: (Fx, Fy, Fz)}"""
        k = self.i % self.N
        self.t[k] = t
        for label, (Fx, Fy, Fz) in values.items():
            self.fx[label][k] = Fx
            self.fy[label][k] = Fy
            self.fz[label][k] = Fz
        self.i += 1

    def refresh(self, t_window: float = 10.0):
        if self.i < 2:
            return
        n = min(self.i, self.N)
        idx = (np.arange(n) + self.i - n) % self.N
        tt = self.t[idx]

        for label, ax in zip(self.labels, self.axes[:len(self.labels)]):
            lx, ly, lz = self.lines[label]
            lx.set_data(tt, self.fx[label][idx])
            ly.set_data(tt, self.fy[label][idx])
            lz.set_data(tt, self.fz[label][idx])

            # keep a sliding time window
            x1 = tt[-1]
            x0 = max(tt[0], x1 - t_window)
            ax.set_xlim(x0, x0 + t_window)

            # symmetric y-limits around 0
            ymax = max(
                1e-3,
                np.nanmax(np.abs(self.fx[label][idx])),
                np.nanmax(np.abs(self.fy[label][idx])),
                np.nanmax(np.abs(self.fz[label][idx])),
            )
            ax.set_ylim(-ymax, ymax)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
