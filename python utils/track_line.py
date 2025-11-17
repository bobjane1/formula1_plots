import numpy as np
import fastf1
import my_f1_utils # cache
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def get_pos_data():
    session = fastf1.get_session(2024, 22, "Q")
    session.load()
    laps = session.laps.split_qualifying_sessions()[2]
    drivers = laps["Driver"].unique()
    ans = []
    for driver in drivers:
        driver_laps = laps.pick_drivers(driver)
        lap = driver_laps.pick_fastest()
        pos_data = lap.get_pos_data(pad=1, pad_side='both')
        lap_coords = [pos_data[c].to_numpy()/10 for c in ["X", "Y", "Z"]]
        ans.append(np.stack(lap_coords, axis=1))
    return ans

def _smooth_lap(points, window_length, polyorder):
    """
    Smooth a single lap (N,3) with a Savitzky-Golay filter.
    """
    points = np.asarray(points)
    n = len(points)
    if n < 5:
        return points

    # window_length must be odd and <= n
    window_length = min(window_length, n - (1 - n % 2))
    if window_length < polyorder + 2:
        window_length = polyorder + 2 + (1 - (polyorder + 2) % 2)

    window_length = min(window_length, n - (1 - n % 2))

    smoothed = np.empty_like(points)
    for dim in range(3):
        smoothed[:, dim] = savgol_filter(points[:, dim],
                                         window_length=window_length,
                                         polyorder=polyorder,
                                         mode="interp")
    return smoothed


def _arc_length_parameterize(points):
    """
    Given (N,3) points, return:
        s_norm: normalized arc-length parameter in [0,1]
    """
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    if s[-1] == 0:
        return s  # degenerate
    s_norm = s / s[-1]
    return s_norm


def _resample_lap(points, s_norm, s_grid):
    """
    Resample one lap onto a common normalized arc-length grid.
    """
    points = np.asarray(points)
    resampled = np.empty((len(s_grid), 3))
    for dim in range(3):
        resampled[:, dim] = np.interp(s_grid, s_norm, points[:, dim])
    return resampled


def fit_track_centerline(
    laps,
    n_centerline_points=500,
    smooth_window=21,
    smooth_polyorder=3,
    spline_smooth=0.5,
):
    """
    Fit a smooth, closed 3D spline representing the track centerline.

    Parameters
    ----------
    laps : list of array-like
        Each element is an array of shape (N_i, 3) with columns (x, y, z)
        for one complete lap in order.
    n_centerline_points : int
        Number of points used to estimate the centerline before spline fit.
    smooth_window : int
        Savitzky-Golay filter window length (samples).
    smooth_polyorder : int
        Savitzky-Golay polynomial order.
    spline_smooth : float
        Smoothing factor for splprep (larger → smoother spline).

    Returns
    -------
    track : callable
        track(t) → (x, y, z) for t in [0, 1).
    tck : tuple
        Spline representation as returned by scipy.interpolate.splprep.
    s_grid : ndarray
        Normalized arc-length grid used to build the centerline.
    centerline_points : ndarray
        Array of shape (n_centerline_points, 3) with the averaged centerline.
    """
    if isinstance(laps, np.ndarray):
        # Allow a single lap as a plain (N,3) array
        laps = [laps]

    # 1. Smooth each lap
    smoothed_laps = [
        _smooth_lap(np.asarray(lap), window_length=smooth_window, polyorder=smooth_polyorder)
        for lap in laps
    ]

    # 2. Parameterize each lap by normalized arc length [0,1]
    s_norm_laps = [_arc_length_parameterize(lap) for lap in smoothed_laps]

    # 3. Resample all laps onto a common arc-length grid
    s_grid = np.linspace(0.0, 1.0, n_centerline_points)
    resampled_laps = [
        _resample_lap(lap, s_norm, s_grid)
        for lap, s_norm in zip(smoothed_laps, s_norm_laps)
    ]

    # 4. Average across laps to get a centerline (approx track center)
    centerline_points = np.mean(np.stack(resampled_laps, axis=0), axis=0)

    # Ensure closed curve by forcing last point equal to first
    if np.linalg.norm(centerline_points[0] - centerline_points[-1]) > 1e-6:
        centerline_points[-1] = centerline_points[0]

    # 5. Fit a periodic B-spline in 3D
    x, y, z = centerline_points.T
    # u is the parameter, tck is the spline; per=True enforces periodicity
    tck, u = splprep([x, y, z], s=spline_smooth, per=True)

    # 6. Build a convenient callable
    def track(t):
        """
        Evaluate the track spline at parameter t in [0,1] (can be vector).
        Returns an array of shape (..., 3).
        """
        t = np.mod(t, 1.0)  # wrap around
        x_t, y_t, z_t = splev(t, tck)
        return np.stack([np.array(x_t), np.array(y_t), np.array(z_t)], axis=-1)

    return track, tck, s_grid, centerline_points


class PanZoomHandler:
    """
    Simple matplotlib event handler that enables mouse drag panning and scroll zooming.
    """

    def __init__(self, ax, base_scale=1.2):
        self.ax = ax
        self.base_scale = base_scale
        self.press = None
        self.cid_scroll = ax.figure.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.cid_press = ax.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = ax.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata = event.xdata if event.xdata is not None else (cur_xlim[0] + cur_xlim[1]) / 2
        ydata = event.ydata if event.ydata is not None else (cur_ylim[0] + cur_ylim[1]) / 2
        scale_factor = 1 / self.base_scale if event.button == "up" else self.base_scale
        width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0]) if cur_xlim[1] != cur_xlim[0] else 0.5
        rely = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0]) if cur_ylim[1] != cur_ylim[0] else 0.5
        self.ax.set_xlim(xdata - relx * width, xdata + (1 - relx) * width)
        self.ax.set_ylim(ydata - rely * height, ydata + (1 - rely) * height)
        self.ax.figure.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        self.press = (event.xdata, event.ydata)
        self.x0, self.x1 = self.ax.get_xlim()
        self.y0, self.y1 = self.ax.get_ylim()

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - self.press[0]
        dy = event.ydata - self.press[1]
        self.ax.set_xlim(self.x0 - dx, self.x1 - dx)
        self.ax.set_ylim(self.y0 - dy, self.y1 - dy)
        self.ax.figure.canvas.draw_idle()

    def on_release(self, event):
        self.press = None

def main():
    laps = get_pos_data()
    track, tck, s_grid, centerline = fit_track_centerline(laps)
    t_plot = np.linspace(0, 1, 1000)
    track_points = track(t_plot)
    fig, ax = plt.subplots()
    for lap in laps:
        ax.plot(lap[:, 0], lap[:, 1], alpha=0.3, label="raw lap" if lap is laps[0] else None)
    ax.plot(centerline[:, 0], centerline[:, 1], "o", ms=3, label="avg centerline pts")
    ax.plot(track_points[:, 0], track_points[:, 1], "-", lw=2, label="fitted spline")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.set_title("Track fitting from noisy GPS laps")
    PanZoomHandler(ax)
    plt.show()


main()
