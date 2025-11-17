import fastf1
import my_f1_utils # cache
import numpy as np
import scipy.signal
import scipy.interpolate
import scipy.spatial   # <-- add this
import matplotlib.pyplot as plt

def get_pos_data():
    session = fastf1.get_session(2024, 22, "Q")
    session.load()
    laps = session.laps.split_qualifying_sessions()[2]
    drivers = laps["Driver"].unique()
    ans = {}
    for driver in drivers:
        driver_laps = laps.pick_drivers(driver)
        # driver_laps = driver_laps[driver_laps["LapNumber"] != 24]
        lap = driver_laps.pick_fastest()
        
        pos_data = lap.get_pos_data(pad=1, pad_side='both')
        pos_coords = [pos_data[c].to_numpy()/10 for c in ["X", "Y", "Z"]]
        pos_t = pos_data["Time"].dt.total_seconds().to_numpy()

        car_data = lap.get_car_data(pad=1, pad_side='both')
        car_t = car_data["Time"].dt.total_seconds().to_numpy()
        car_speed = car_data["Speed"].to_numpy()
        
        # if driver == "RUS":
            # print(lap["LapNumber"])
            # print(f"{driver}|{lap['LapTime'].total_seconds()}|{lap['Sector1Time'].total_seconds()}|{lap['Sector2Time'].total_seconds()}|{lap['Sector3Time'].total_seconds()}")
            # for row in car_data.itertuples():
                # print(f"{row.Time.total_seconds()}|{row.Speed}|{row.Throttle}|{row.Brake}|{row.nGear}|{row.RPM}|{row.DRS}")

        ans[driver] = {
            "pos_data": {"coords": np.stack(pos_coords, axis=1), "times": pos_t},
            "car_data": {"speed": car_speed, "times": car_t},
        }
    return ans

def _smooth_lap(points, window_length, polyorder):
    assert len(points) > 21 # arbitrary
    assert window_length % 2 == 1
    assert window_length <= len(points)
    assert (polyorder+2) < window_length

    smoothed = np.empty_like(points)
    for dim in range(3):
        smoothed[:, dim] = scipy.signal.savgol_filter(points[:, dim],
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
    n_centerline_points=1000,
    smooth_window=11,
    smooth_polyorder=7,
    spline_smooth=0.1,
):
    """
    Fit a smooth, closed 3D spline representing the track centerline.
    """
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

    # Ensure closed curve by forcing last point equal to first (averaged)
    centerline_points[-1] = (centerline_points[-1] + centerline_points[0]) / 2
    centerline_points[0] = centerline_points[-1]

    # 5. Fit a periodic B-spline in 3D
    x, y, z = centerline_points.T
    tck, u = scipy.interpolate.splprep([x, y, z], s=spline_smooth, per=True)

    # 6. Build a convenient callable
    def track(t):
        """
        Evaluate the track spline at parameter t in [0,1] (can be vector).
        Returns an array of shape (..., 3).
        """
        t = np.mod(t, 1.0)  # wrap around
        x_t, y_t, z_t = scipy.interpolate.splev(t, tck)
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

def make_plot(lines):
    fig, ax = plt.subplots()
    for label, line in lines.items():
        ax.plot(line["vals"][:, 0], line["vals"][:, 1], label=label, **line["options"])
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.set_title("Track fitting from noisy GPS laps")
    PanZoomHandler(ax)
    plt.show()

def main():
    laps = get_pos_data()
    track, tck, s_grid, centerline = fit_track_centerline([x["pos_data"]["coords"] for x in laps.values()])

    # dense sample of the spline to build a KD-tree for fast projection
    t_plot = np.linspace(0, 1, 10000)  # denser sample for better projection accuracy
    track_points = track(t_plot)

    # KD-tree on the spline samples (3D)
    tree = scipy.spatial.cKDTree(track_points)

    lines = {}
    for driver, lap_info in laps.items():
        if driver != "SAI": continue

        lines[f"{driver}1"] = {
            "vals": lap_info["pos_data"]["coords"],
            "options": {"alpha": 0.5, "marker": "o", "ms": 2}
        }

        # projected_lap: nearest spline sample for each lap point
        dists, idx = tree.query(lap_info["pos_data"]["coords"], k=1)   # lap shape (N,3)
        projected_lap = track_points[idx]   # shape (N,3)

        lines[f"{driver}2"] = {
            "vals": projected_lap,
            "options": {"alpha": 0.5, "marker": "x", "ms": 2}
        }

        # for li in range(len(lap["coords"])):
            # print(f"{driver}|{lap['times'][li]}|{idx[li]}|{projected_lap[li,0]}|{projected_lap[li,1]}|{projected_lap[li,2]}|{lap['coords'][li,0]}|{lap['coords'][li,1]}|{lap['coords'][li,2]}|{dists[li]}")

        car_data_coords = []
        for idx in range(len(lap_info["car_data"]["times"])):
            t_car = lap_info["car_data"]["times"][idx]
            coords = np.apply_along_axis(lambda col: np.interp(t_car, lap_info["pos_data"]["times"], col), 0, lap_info["pos_data"]["coords"])
            car_data_coords.append(coords)
            # print(f"{driver}|{t_car}|{coords[0]}|{coords[1]}|{coords[2]}|{lap_info['car_data']['speed'][idx]}")
        dists, idxes = tree.query(car_data_coords, k=1)

        for i,idx in enumerate(idxes):            
            speed = lap_info["car_data"]["speed"][i]
            print(f"{idx}|{speed}")

    lines["fitted spline"] = {
        "vals": track_points,
        "options": {"linestyle": "-", "lw": 2, "color": "black", "marker": "o", "ms": 3}
    }

    # make_plot(lines)

main()
