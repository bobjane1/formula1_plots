import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import fastf1
from fastf1 import plotting as f1plot
import scipy.optimize
import my_f1_utils # cache

def interpolate_pos_data(times, lap):
    pos_data = lap.get_pos_data()
    pos_data["Seconds"] = pos_data["Time"].dt.total_seconds()
    xs = pos_data["X"].to_numpy()/10
    ys = pos_data["Y"].to_numpy()/10
    origin = ((xs[0]+xs[-1])/2, (ys[0]+ys[-1])/2)
    xs = np.concatenate(([origin[0]], xs,[origin[0]]))
    ys = np.concatenate(([origin[1]], ys,[origin[1]]))
    t1 = pos_data["Seconds"].to_numpy()
    t1 = np.concatenate(([0.0], t1, [lap["LapTime"].total_seconds()]))

    car_data = lap.get_car_data()
    car_data["Seconds"] = car_data["Time"].dt.total_seconds()

    segment_distances = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    t_xy_speeds = np.interp((t1[:-1] + t1[1:]) / 2, car_data["Seconds"].to_numpy(), car_data["Speed"].to_numpy())

    homemade = True
    if not homemade:
        def err_func(t_xy_adj, segment_distances, desired_speeds):
            t_deltas = np.diff(t_xy_adj)
            if np.any(t_deltas <= 0): return 1e6 # penalize heavily
            implied_speeds = segment_distances / t_deltas * 3.6
            speed_error = np.mean((implied_speeds - desired_speeds)**2)
            return speed_error
        
        result = scipy.optimize.minimize(
            err_func, t1, args=(segment_distances, t_xy_speeds),
            method='L-BFGS-B', options={'maxiter': 100000, "maxfun": 100000}
        )

        return np.interp(times, result.x, xs), np.interp(times, result.x, ys)
    else:
        tadj = t1.copy()
        lr = 0.01
        last_err = 1e6
        for idx in range(10000):
            t_deltas = np.diff(tadj)
            implied_speeds = segment_distances / t_deltas * 3.6
            speed_error = implied_speeds - t_xy_speeds
            grad1 = 2*speed_error*3.6*segment_distances/t_deltas**2 / len(implied_speeds) # derivative
            grads = np.concatenate(([grad1[0]], grad1[1:]-grad1[:-1], [-grad1[-1]]))
            tot_err = np.mean(speed_error**2)
            if tot_err > last_err: lr *= 0.1
            last_err = tot_err
            tadj -= min(lr, lr / (np.abs(grads).max()+1e-6)) * grads

        return np.interp(times, tadj, xs), np.interp(times, tadj, ys)

def calc_time_delta(ts, x1, y1, x2, y2):
    ans = []
    for ti, t in enumerate(ts):
        dists_sq = (x2 - x1[ti])**2 + (y2 - y1[ti])**2
        min_idx = np.argmin(dists_sq)
        ans.append(t - ts[min_idx])
    return np.array(ans)


def make_plot(laps, labels, colors, title, fps = 60, dpi = 200):
    t_max = 0.0
    for lap in laps:
        t_max = max(t_max, lap["LapTime"].total_seconds())

    n_frames = max(2, int(np.ceil(t_max * fps)))
    T = np.linspace(0.0, t_max, n_frames)

    all_xs, all_ys = [], []
    dots, delta_lines = [], []
    time_deltas = []

    # Two panels: left = track, right = time-delta
    fig, (ax, ax_td) = plt.subplots(1, 2, figsize=(6, 3), dpi=dpi, gridspec_kw={'width_ratios': [2, 1]})
    f1plot.setup_mpl()  # optional styling

    for di, driver in enumerate(labels):
        driver_lap = laps[di]
        xi, yi = interpolate_pos_data(T, driver_lap)
        # return
        all_xs.append(xi)
        all_ys.append(yi)

        # Moving dot on the track
        dot, = ax.plot([], [], "o", ms=4, label=driver, color=colors[di], zorder=3)
        dots.append(dot)

        # calc the gap to the first driver
        if di > 0:
            td = calc_time_delta(T, xi, yi, all_xs[0], all_ys[0])
            time_deltas.append(td)
            ln, = ax_td.plot([], [], lw=1, color=colors[di], label=driver)
            delta_lines.append(ln)

    # ===== Track axis (left) =====
    ax.set_aspect("equal", adjustable="box")
    track_xs = np.concatenate(all_xs)
    track_ys = np.concatenate(all_ys)
    mx, my = track_xs.mean(), track_ys.mean()
    base_r = max(track_xs.max() - track_xs.min(), track_ys.max() - track_ys.min()) * 0.55

    # Zoom state
    zoom_state = {'level': 0.8}

    ax.set_xlim(mx - base_r, mx + base_r)
    ax.set_ylim(my - base_r, my + base_r)
    ax.set_axis_off()
    ax.plot(track_xs, track_ys, lw=2, alpha=0.25)

    ax.set_title(title, fontsize=8, pad=10)
    clock_txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=8)
    speed_txt = ax.text(0.02, 0.92, "", transform=ax.transAxes, va="top", ha="left", fontsize=6)
    zoom_txt = ax.text(0.98, 0.98, "Zoom: 100% [+/-]", transform=ax.transAxes, va="top", ha="right", fontsize=6)
    zoom_txt.set_text(f"Zoom: {100/zoom_state['level']:.0f}% [+/-]")

    ax.legend(loc="lower right", fontsize=4)

    # ===== Time-delta axis (right) =====
    ax_td.set_title(f"Gap")
    ax_td.xaxis.set_ticklabels([])
    ax_td.set_xlim(0, T[-1])
    ax_td.set_ylim(-0.5, 2)  # 0s to 2s
    ax_td.grid(True, alpha=0.3)
    ax_td.axhline(0, lw=1, alpha=0.6)

    playback_state = {"frame_idx": 0.0, "speed": 1.0, "playing": True}
    max_speed = 10.0
    speed_step = 1.0

    fig.subplots_adjust(bottom=0.18)

    button_height = 0.05
    button_width = 0.12
    button_gap = 0.02
    button_y = 0.03
    button_x = 0.1
    play_ax = fig.add_axes([button_x, button_y, button_width, button_height])
    pause_ax = fig.add_axes([button_x + button_width + button_gap, button_y, button_width, button_height])
    faster_ax = fig.add_axes([button_x + 2 * (button_width + button_gap), button_y, button_width, button_height])
    slower_ax = fig.add_axes([button_x + 3 * (button_width + button_gap), button_y, button_width, button_height])

    play_btn = Button(play_ax, "Play")
    pause_btn = Button(pause_ax, "Pause")
    faster_btn = Button(faster_ax, "Faster")
    slower_btn = Button(slower_ax, "Slower")

    def update_zoom_text():
        zoom_txt.set_text(f"Zoom: {100/zoom_state['level']:.0f}% [+/-]")
        fig.canvas.draw_idle()

    def update_viewport(frame_idx):
        """Update viewport to keep drivers in frame, prioritizing front runner"""
        current_xs = [all_xs[di][frame_idx] for di in range(len(labels))]
        current_ys = [all_ys[di][frame_idx] for di in range(len(labels))]

        front_x, front_y = current_xs[0], current_ys[0]
        min_x, max_x = min(current_xs), max(current_xs)
        min_y, max_y = min(current_ys), max(current_ys)

        padding = 50  # meters
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding

        all_cx = (min_x + max_x) / 2
        all_cy = (min_y + max_y) / 2
        all_r = max(max_x - min_x, max_y - min_y) / 2

        target_r = base_r * zoom_state['level']

        if all_r > target_r:
            cx, cy = front_x, front_y
            r = target_r
        else:
            cx, cy = all_cx, all_cy
            r = max(all_r, target_r * 0.5)

        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)

    def on_key(event):
        """Handle zoom controls"""
        if event.key == '+' or event.key == '=':
            zoom_state['level'] *= 0.8
            zoom_state['level'] = max(0.025, zoom_state['level'])
        elif event.key == '-' or event.key == '_':
            zoom_state['level'] *= 1.25
            zoom_state['level'] = min(5.0, zoom_state['level'])
        zoom_state['level'] = 1.0 / round(1.0 / zoom_state['level'], 1)
        update_zoom_text()

    fig.canvas.mpl_connect('key_press_event', on_key)

    def set_play(_):
        playback_state["playing"] = True

    def set_pause(_):
        playback_state["playing"] = False

    def speed_up(_):
        playback_state["speed"] = min(max_speed, playback_state["speed"] + speed_step)
        speed_txt.set_text(f"Speed: {playback_state['speed']:+.1f}x")
        fig.canvas.draw_idle()

    def slow_down(_):
        playback_state["speed"] = max(-max_speed, playback_state["speed"] - speed_step)
        speed_txt.set_text(f"Speed: {playback_state['speed']:+.1f}x")
        fig.canvas.draw_idle()

    play_btn.on_clicked(set_play)
    pause_btn.on_clicked(set_pause)
    faster_btn.on_clicked(speed_up)
    slower_btn.on_clicked(slow_down)

    def init():
        playback_state["frame_idx"] = 0.0
        playback_state["playing"] = True
        for dot in dots:
            dot.set_data([], [])
        for ln in delta_lines:
            ln.set_data([], [])
        clock_txt.set_text("")
        speed_txt.set_text(f"Speed: {playback_state['speed']:+.1f}x")
        return (*dots, *delta_lines, clock_txt, speed_txt, zoom_txt)

    def update(_):
        playback_state["frame_idx"] = max(0.0, min(n_frames - 1, playback_state["frame_idx"]))
        frame_idx = int(round(playback_state["frame_idx"]))

        for di, dot in enumerate(dots):
            dot.set_data([all_xs[di][frame_idx]], [all_ys[di][frame_idx]])
        for di, ln in enumerate(delta_lines):
            ln.set_data(T[:frame_idx+1], time_deltas[di][:frame_idx+1])

        clock_txt.set_text(f"t = {T[frame_idx]:.2f} s")
        speed_txt.set_text(f"Speed: {playback_state['speed']:+.1f}x")
        update_viewport(frame_idx)

        if playback_state["playing"]:
            playback_state["frame_idx"] += playback_state["speed"]
        playback_state["frame_idx"] = max(0.0, min(n_frames - 1, playback_state["frame_idx"]))

        return (*dots, *delta_lines, clock_txt, speed_txt, zoom_txt)

    anim = FuncAnimation(fig, update, init_func=init, interval=1000/fps, repeat=True, blit=False)
    plt.show()
    plt.close(fig)

def animate_q3_fastest_laps(year, grand_prix, drivers):
    session = fastf1.get_session(year, grand_prix, "Q")
    session.load()
    laps = session.laps.split_qualifying_sessions()[2]

    driver_laps = []
    colors = []
    title =  f"{session.event['EventName']} {year} – Q3 Fastest Laps"
    for driver in drivers:    
        driver_laps.append(laps.pick_drivers(driver).pick_fastest())
        color = fastf1.plotting.get_driver_style(identifier=driver, style=['color'], session=session)["color"]
        if driver == "PIA":
            color = "black"
        colors.append(color)
    make_plot(driver_laps, drivers, colors, title)

animate_q3_fastest_laps(2025, 24, ["VER", "NOR"])

# year, grand_prix = 2025, 23
# session = fastf1.get_session(year, grand_prix, "Q")
# session.load()
# laps = session.laps.split_qualifying_sessions()[2]
# lap = laps[(laps["Driver"] == "NOR") & (laps["LapNumber"] == 17)].iloc[0]
# car_data = lap.get_car_data()
# for row in car_data.itertuples():
#     print(f"{row.Time.total_seconds()}|{row.Speed}|{row.Throttle}|{row.Brake}")

# pos_data = lap.get_pos_data()
# for row in pos_data.itertuples():
#     print(f"{row.Time.total_seconds()}|{row.X}|{row.Y}")