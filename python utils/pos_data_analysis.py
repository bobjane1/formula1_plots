import numpy as np
import fastf1
import my_f1_utils # cache


def optimize_t(raw_ts, xs, ys, speeds):
    segment_distances = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    tadj = raw_ts.copy()
    lr = 0.01
    last_err = 1e6
    for idx in range(1000):
        t_deltas = np.diff(tadj)
        implied_speeds = segment_distances / t_deltas * 3.6
        speed_error = implied_speeds - speeds
        grad1 = 2*speed_error*3.6*segment_distances/t_deltas**2 / len(implied_speeds) # derivative
        grads = np.concatenate(([grad1[0]], grad1[1:]-grad1[:-1], [-grad1[-1]]))
        tot_err = np.mean(speed_error**2)
        if tot_err > last_err: lr *= 0.5
        last_err = tot_err
        # if idx % 10000 == 0:
            # print(f"{idx//10000}|{lr}|{last_err}")
        tadj -= min(lr, lr / (np.abs(grads).max()+1e-6)) * grads
    return tadj

def interpolated_coordinates(driver_laps, lap_num):
    lap = driver_laps[driver_laps["LapNumber"] == lap_num]
    pos_data = lap.get_pos_data(pad=1, pad_side='both')
    pos_data["Seconds"] = pos_data["Time"].dt.total_seconds()
    pos_data["Session_Seconds"] = pos_data["SessionTime"].dt.total_seconds()
    pos_data["x_m"] = pos_data["X"].to_numpy()/10
    pos_data["y_m"] = pos_data["Y"].to_numpy()/10
        
    # for row in pos_data.itertuples():
        # print(f"{row.SessionTime.total_seconds()}|{row.Time.total_seconds()}|{row.X}|{row.Y}|{row.Z}")

    car_data = lap.get_car_data(pad=1, pad_side='both')
    car_data["Seconds"] = car_data["Time"].dt.total_seconds()
    car_data["Session_Seconds"] = car_data["SessionTime"].dt.total_seconds()
    for row in car_data.itertuples():
        print(f"{row.SessionTime.total_seconds()}|{row.Time.total_seconds()}|{row.RPM}|{row.Speed}|{row.nGear}|{row.Throttle}|{row.Brake}|{row.DRS}")
    
    xs = pos_data["x_m"].to_numpy()
    ys = pos_data["y_m"].to_numpy()
    ts = pos_data["Seconds"].to_numpy()
    
    car_seconds = car_data["Seconds"].to_numpy()
    car_speeds = car_data["Speed"].to_numpy()

    t_xy_speeds = np.interp((ts[:-1] + ts[1:]) / 2, car_seconds, car_speeds)
    segment_distances = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    dTs = (segment_distances / t_xy_speeds * 3.6)
    sum_blocks = dTs.sum() + ts[0]

    actual = ts[-1]
    # print(f"{lap_num}|{sum_blocks}|{actual}|{actual-sum_blocks}")
    
    tadj = optimize_t(ts, xs, ys, t_xy_speeds)
    # print("T|X|Y|S|TADJ")
    # for i in range(len(ts)):
    #     if i == len(ts)-1:
    #         print(f"{ts[i]}|{xs[i]}|{ys[i]}|0|{tadj[i]}")
    #     else:
    #         print(f"{ts[i]}|{xs[i]}|{ys[i]}|{t_xy_speeds[i]}|{tadj[i]}")

    # print("T|S")
    # for i in range(len(car_seconds)):
    #     print(f"{car_seconds[i]}|{car_speeds[i]}")

def find_breaking_point(driver, driver_laps, lap_num):
    lap = driver_laps[driver_laps["LapNumber"] == lap_num]
    pos_data = lap.get_pos_data(pad=1, pad_side='both')
    pt = pos_data["Time"].dt.total_seconds().to_numpy()
    px = pos_data["X"].to_numpy()/10
    py = pos_data["Y"].to_numpy()/10


    car_data = lap.get_car_data(pad=1, pad_side='both')
    car_data["Seconds"] = car_data["Time"].dt.total_seconds()
    # start breaking times. Points where the previous row has Brake==False and the current row has Brake==True
    brake_starts = car_data[(car_data["Brake"].shift(2) == 0) & (car_data["Brake"].shift(1) == 0) & (car_data["Brake"] == 1) & (car_data["Brake"].shift(-1) == 1)]
    for row in brake_starts.itertuples():
        rt = row.Time.total_seconds()
        if abs(rt-20) > 5: continue
        inter_x = np.interp(row.Seconds, pt, px)
        inter_y = np.interp(row.Seconds, pt, py)
        print(f"{driver}|{rt}|{inter_x}|{inter_y}")

    brake_ends = car_data[(car_data["Brake"].shift(2) == 1) & (car_data["Brake"].shift(1) == 1) & (car_data["Brake"] == 0) & (car_data["Brake"].shift(-1) == 0)]
    for row in brake_ends.itertuples():
        rt = row.Time.total_seconds()
        if abs(rt-22) > 5: continue
        inter_x = np.interp(row.Seconds, pt, px)
        inter_y = np.interp(row.Seconds, pt, py)
        print(f"{driver}|{rt}|{inter_x}|{inter_y}")

session = fastf1.get_session(2024, 22, "Q")
session.load()

laps = session.laps.split_qualifying_sessions()[2]
drivers = laps["Driver"].unique()
for driver in drivers:
    if driver != "LEC": continue
    # print(driver)
    
    
    driver_laps = laps.pick_drivers(driver)
    # for lap in driver_laps.iterlaps():
    #     print(f"{lap[1]['LapNumber']}|{lap[1]['LapTime'].total_seconds()}")

    # ver
    lap_num = 17
    lap_num = 20

    # rus
    # lap_num = 21
    # lap_num = 24
    lap_num = driver_laps.pick_fastest()["LapNumber"]
    
    # interpolated_coordinates(driver_laps, lap_num)
    find_breaking_point(driver, driver_laps, lap_num)
