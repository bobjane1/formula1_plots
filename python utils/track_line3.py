import fastf1
import my_f1_utils # cache
import numpy as np
import scipy.signal
import scipy.interpolate
import scipy.spatial 
import matplotlib.pyplot as plt

def get_pos_data():
    session = fastf1.get_session(2024, 22, "Q")
    session.load()
    laps = session.laps.split_qualifying_sessions()[2]
    drivers = laps["Driver"].unique()
    ans = {}
    for driver in drivers:
        driver_laps = laps.pick_drivers(driver)
        lap = driver_laps.pick_fastest()
        
        pos_data = lap.get_pos_data(pad=1, pad_side='both')
        pos_coords = [pos_data[c].to_numpy()/10 for c in ["X", "Y", "Z"]]
        pos_t = pos_data["Time"].dt.total_seconds().to_numpy()

        car_data = lap.get_car_data(pad=1, pad_side='both')
        car_t = car_data["Time"].dt.total_seconds().to_numpy()
        car_speed = car_data["Speed"].to_numpy()

        ans[driver] = {
            "pos_data": {"coords": np.stack(pos_coords, axis=1), "times": pos_t},
            "car_data": {"speed": car_speed, "times": car_t},
            "sector_times": [lap[x].total_seconds() for x in ["Sector1Time", "Sector2Time", "Sector3Time"]],
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

def make_plot(lines):
    fig, ax = plt.subplots()
    for label, line in lines.items():
        ax.plot(line["vals"][:, 0], line["vals"][:, 1], label=label, **line["options"])
    ax.set_aspect("auto")
    ax.legend()
    ax.set_title("Track fitting from noisy GPS laps")
    plt.show()

def main():
    laps = get_pos_data()
    track, tck, s_grid, centerline = fit_track_centerline([x["pos_data"]["coords"] for x in laps.values()])

    t_plot = np.linspace(0, 1, 10000)  # denser sample for better projection accuracy
    track_points = track(t_plot)
    tree = scipy.spatial.cKDTree(track_points)

    session = fastf1.get_session(2024, 22, "Q")
    session.load()
    laps = session.laps.split_qualifying_sessions()[2]
    laps = session.laps
    drivers = laps["Driver"].unique()
    for driver in drivers:
        # if driver != "RUS": continue
        driver_laps = laps.pick_drivers(driver)
        for li,lap in driver_laps.iterrows():
            pos_data = lap.get_pos_data(pad=1, pad_side='both')
            pos_coords = [pos_data[c].to_numpy()/10 for c in ["X", "Y", "Z"]]
            pos_t = pos_data["Time"].dt.total_seconds().to_numpy()
            sec_times = [lap[x].total_seconds() for x in ["Sector1Time", "Sector2Time", "Sector3Time"]]
            if any(np.isnan(x) for x in sec_times): continue
            print(f"{driver}|{lap['LapNumber']}", end="|")
            for si in range(3):
                sec_t = sum(sec_times[:si+1])
                # interpolate on lap info pos_data
                interp_pos = [np.interp(sec_t, pos_t, pos_coords[x]) for x in range(3)]
                print("|".join(map(str, interp_pos)), end="|")
                dist_to_track, idx = tree.query(interp_pos)
                
                if si==2 and idx < 9000: idx += len(track_points)
                print(f"{sec_t}|{idx}|{dist_to_track}",end="|")
            print()

main()