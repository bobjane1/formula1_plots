import fastf1
import my_f1_utils # cache
import numpy as np
import scipy.signal
import scipy.interpolate
import scipy.spatial 
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Callable
from typing import List
from scipy.interpolate import PchipInterpolator

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
        #     print(f"{row.Time.total_seconds()}|{row.Speed}|{row.Throttle}|{row.Brake}|{row.nGear}|{row.RPM}|{row.DRS}")

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

def make_plot(lines):
    fig, ax = plt.subplots()
    for label, line in lines.items():
        ax.plot(line["vals"][:, 0], line["vals"][:, 1], label=label, **line["options"])
    ax.set_aspect("auto")
    ax.legend()
    ax.set_title("Track fitting from noisy GPS laps")
    plt.show()

# ====== TRYING DIFFERENT MODELS

@dataclass
class SpeedSegment:
    s0: float
    s1: float
    interp: PchipInterpolator


def _find_breakpoints_raw(s, v, grad_frac=0.4, min_seg_len=10):
    """
    Find indices where we should split the profile based on large |dv/ds|
    computed directly from the raw signal.
    grad_frac is a fraction of max(|dv/ds|).
    """
    s = np.asarray(s)
    v = np.asarray(v)
    N = len(s)

    # numeric gradient wrt s on raw data
    dv_ds = np.gradient(v, s)
    abs_grad = np.abs(dv_ds)

    gmax = abs_grad.max()
    if gmax <= 0:
        return [0, N - 1]  # degenerate, flat trace

    thr = grad_frac * gmax

    # peaks in |dv/ds| correspond to strong braking/accel events
    peaks, _ = scipy.signal.find_peaks(abs_grad, height=thr, distance=min_seg_len)

    # turn peaks into break indices
    break_idx = [0]
    for idx in peaks:
        if idx - break_idx[-1] >= min_seg_len:
            break_idx.append(int(idx))

    # ensure we end at the last sample
    if N - 1 - break_idx[-1] >= min_seg_len:
        break_idx.append(N - 1)
    elif break_idx[-1] != N - 1:
        break_idx.append(N - 1)

    break_idx = sorted(set(break_idx))
    return break_idx


def fit_speed_profile_piecewise_pchip(
    t_track,
    v_car,
    grad_frac=0.4,
    min_seg_len=10,
):
    """
    Fit a piecewise, shape-preserving speed profile v(t) by:
      - computing dv/dt on the *raw* speed trace,
      - splitting at large |dv/dt| (braking / big accel),
      - fitting PCHIP (monotone cubic Hermite) per segment.

    Parameters
    ----------
    t_track : array-like, shape (N,)
        Monotonic lap parameter (e.g. 0..1 or sample index).
    v_car : array-like, shape (N,)
        Speed at each parameter sample.
    grad_frac : float in (0,1)
        Fraction of max(|dv/dt|) used as threshold to detect split points.
        Larger -> fewer segments; smaller -> more segments.
    min_seg_len : int
        Minimum number of samples per segment.

    Returns
    -------
    speed_model : callable
        speed_model(t_query) -> fitted speed at those lap parameters.
    segments : list[SpeedSegment]
        Fitted segments with their [s0, s1] and PCHIP interpolators.
    """
    t_track = np.asarray(t_track, dtype=float)
    v_car = np.asarray(v_car, dtype=float)
    assert t_track.shape == v_car.shape
    N = len(t_track)
    if N < 5:
        raise ValueError("Not enough points to fit a profile")

    # 1. Normalize parameter to [0,1] for numerical stability
    t_min, t_max = t_track[0], t_track[-1]
    s = (t_track - t_min) / (t_max - t_min + 1e-12)  # s in [0,1]

    # 2. Find breakpoints *directly on raw speed*
    break_idx = _find_breakpoints_raw(s, v_car, grad_frac=grad_frac, min_seg_len=min_seg_len)

    # 3. Fit PCHIP in each segment
    segments: List[SpeedSegment] = []
    for i in range(len(break_idx) - 1):
        i0 = break_idx[i]
        i1 = break_idx[i + 1]
        if i1 <= i0 + 1:
            continue  # too short

        # inside your loop over segments
        s_seg = s[i0:i1 + 1]
        v_seg = v_car[i0:i1 + 1]

        s_seg_u, v_seg_u = deduplicate_x(s_seg, v_seg, reducer="mean")
        if len(s_seg_u) < 2:
            continue  # not enough points after dedup

        # PCHIP is shape-preserving and can't overshoot between points
        interp = PchipInterpolator(s_seg_u, v_seg_u)
        segments.append(SpeedSegment(float(s_seg[0]), float(s_seg[-1]), interp))

    if not segments:
        # Fallback: single global segment
        interp = PchipInterpolator(s, v_car)
        segments = [SpeedSegment(0.0, 1.0, interp)]

    segments.sort(key=lambda seg: seg.s0)
    # Force exact coverage [0,1] on boundaries
    segments[0].s0 = 0.0
    segments[-1].s1 = 1.0

    # Precompute segment edges for fast lookup
    s_edges = np.array([segments[0].s0] + [seg.s1 for seg in segments], dtype=float)

    def speed_model(t_query):
        """
        Evaluate fitted speed profile at arbitrary t_query (scalar or array).
        t_query is in the same units/range as t_track.
        """
        tq = np.asarray(t_query, dtype=float)
        sq = (tq - t_min) / (t_max - t_min + 1e-12)
        sq = np.clip(sq, 0.0, 1.0)

        flat = sq.ravel()
        out = np.empty_like(flat)

        # which segment does each point fall into?
        idx_seg = np.searchsorted(s_edges, flat, side="right") - 1
        idx_seg = np.clip(idx_seg, 0, len(segments) - 1)

        for k, seg in enumerate(segments):
            mask = idx_seg == k
            if not np.any(mask):
                continue
            out[mask] = seg.interp(flat[mask])

        return out.reshape(sq.shape)

    return speed_model, segments

def deduplicate_x(x, y, reducer="mean"):
    """
    Given x, y with possibly repeated x-values, return x_unique, y_agg
    with strictly increasing x suitable for PchipInterpolator.

    reducer: "mean" or "median"
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # sort by x so equal values are adjacent
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # unique x and inverse index mapping
    x_unique, inv = np.unique(x_sorted, return_inverse=True)

    if reducer == "mean":
        # aggregate with bincount for speed
        sums = np.bincount(inv, weights=y_sorted)
        counts = np.bincount(inv)
        y_agg = sums / counts
    else:  # "median" (slower, but more robust)
        y_agg = np.empty_like(x_unique)
        for i in range(len(x_unique)):
            y_agg[i] = np.median(y_sorted[inv == i])

    return x_unique, y_agg


def main():
    laps = get_pos_data()
    track, tck, s_grid, centerline = fit_track_centerline([x["pos_data"]["coords"] for x in laps.values()])

    # dense sample of the spline to build a KD-tree for fast projection
    t_plot = np.linspace(0, 1, 10000)  # denser sample for better projection accuracy
    track_points = track(t_plot)

    # KD-tree on the spline samples (3D)
    tree = scipy.spatial.cKDTree(track_points)

    speed_profiles = {}
    lines = {}
    for driver, lap_info in laps.items():
        if driver not in ["RUS", "SAI", "PIA"]: continue
        if driver == "HAM": continue

        # lines[f"{driver}1"] = {
        #     "vals": lap_info["pos_data"]["coords"],
        #     "options": {"alpha": 0.5, "marker": "o", "ms": 2}
        # }

        # projected_lap: nearest spline sample for each lap point
        dists, idx = tree.query(lap_info["pos_data"]["coords"], k=1)   # lap shape (N,3)
        # projected_lap = track_points[idx]   # shape (N,3)

        # lines[f"{driver}2"] = {
        #     "vals": projected_lap,
        #     "options": {"alpha": 0.5, "marker": "x", "ms": 2}
        # }

        # for li in range(len(lap["coords"])):
            # print(f"{driver}|{lap['times'][li]}|{idx[li]}|{projected_lap[li,0]}|{projected_lap[li,1]}|{projected_lap[li,2]}|{lap['coords'][li,0]}|{lap['coords'][li,1]}|{lap['coords'][li,2]}|{dists[li]}")

        car_data_coords = []
        for idx in range(len(lap_info["car_data"]["times"])):
            t_car = lap_info["car_data"]["times"][idx]
            coords = np.apply_along_axis(lambda col: np.interp(t_car, lap_info["pos_data"]["times"], col), 0, lap_info["pos_data"]["coords"])
            car_data_coords.append(coords)
            # print(f"{driver}|{t_car}|{coords[0]}|{coords[1]}|{coords[2]}|{lap_info['car_data']['speed'][idx]}")
        
        dists, idxes = tree.query(car_data_coords, k=1)

        speed_profiles[driver] = {
            "param": np.array([i/(len(t_plot)-1) for i in idxes]),
            "speed": lap_info["car_data"]["speed"],
        }
        sorted_indices = np.argsort(speed_profiles[driver]["param"])
        speed_profiles[driver]["param"] = speed_profiles[driver]["param"][sorted_indices]
        speed_profiles[driver]["speed"] = speed_profiles[driver]["speed"][sorted_indices]

        acc = np.gradient(lap_info["car_data"]["speed"], lap_info["car_data"]["times"])/3.6/9.81  # g's
        lines[driver] = {
            # "vals": np.column_stack([speed_profiles[driver]["param"], speed_profiles[driver]["speed"]]),
            # "vals": np.column_stack([lap_info["car_data"]["times"], lap_info["car_data"]["speed"]]),
            "vals": np.column_stack([lap_info["car_data"]["times"], acc]),
            "options": {"alpha": 0.7, "marker": "o", "ms": 3}
        }

        # sort

        # for i,idx in enumerate(idxes):            
        #     speed = lap_info["car_data"]["speed"][i]
        #     print(f"{idx}|{speed}")

    # lines["fitted spline"] = {
    #     "vals": track_points,
    #     "options": {"linestyle": "-", "lw": 2, "color": "black", "marker": "o", "ms": 3}
    # }

    make_plot(lines)

    # speed_profile = speed_profiles["RUS"]
    # params = speed_profile["param"]
    # speeds = speed_profile["speed"]

    params = np.concatenate([x["param"] for x in speed_profiles.values()])
    speeds = np.concatenate([x["speed"] for x in speed_profiles.values()])

    sorted_indices = np.argsort(params)
    params = params[sorted_indices]
    speeds = speeds[sorted_indices]
    
    # speed_model, segments = fit_speed_profile(speed_profile["param"], speed_profile["speed"], smooth_window=21, smooth_polyorder=3, prominence=2.0)
    # speed_model, spline = fit_speed_profile_spline(speed_profile["param"], speed_profile["speed"], smooth_factor=1e-3, weight_accel=3.0)


    # speed_model, segments = fit_speed_profile_piecewise_pchip(
    #     params,
    #     speeds,
    #     grad_frac=0.6,   # tweak this to change how aggressively you split
    #     min_seg_len=15,  # avoid tiny segments
    # )

    # s_dense = np.linspace(0, 1, 3000)
    # v_fit = speed_model(s_dense)

    # plt.figure(figsize=(10, 4))
    # plt.plot(params, speeds, alpha=0.7, label="noisy data")    
    # plt.plot(s_dense, v_fit, lw=2, label="fitted profile")
    # plt.xlabel("lap parameter s")
    # plt.ylabel("speed")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()

main()