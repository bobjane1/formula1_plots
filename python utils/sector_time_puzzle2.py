import fastf1, logging, numpy as np, matplotlib.pyplot as plt, my_f1_utils
fastf1.Cache.enable_cache("fastf1_cache")
fastf1.logger.set_log_level(logging.ERROR)
fastf1.Cache.offline_mode(True)

for year in range(2025,2017,-1):
    for round_no in range(1,26):
        try:
            avg_coords = {}
            avg_speeds = {}
            for i, sess_type in enumerate(["Q","R"]):
                session = fastf1.get_session(year, round_no, sess_type)
                session.load()
                circuit_name = session.session_info["Meeting"]["Circuit"]["ShortName"]
                coords = [[],[],[]]
                speeds = [[],[],[]]
                fastest_laptime = session.laps.pick_fastest()["LapTime"].total_seconds()
                for _,lap in session.laps.iterrows():
                    pos_data = lap.get_pos_data(pad=2, pad_side='both')
                    sts = [
                        lap["Sector1Time"].total_seconds(),
                        lap["Sector1Time"].total_seconds()+lap["Sector2Time"].total_seconds(),
                        lap["Sector1Time"].total_seconds()+lap["Sector2Time"].total_seconds()+lap["Sector3Time"].total_seconds(),
                    ]
                    if any(np.isnan(st) for st in sts): continue
                    
                    # interp_t = lambda t: [np.interp(t, pos_data["Time"].dt.total_seconds(), pos_data[c]/10) for c in ["X","Y","Z"]]
                    
                    if lap["LapTime"].total_seconds() > fastest_laptime * 1.07:
                        continue
                    car_data = lap.get_car_data(pad=2, pad_side='both')
                    for i in range(3):
                        coords[i].append([np.interp(sts[i], pos_data["Time"].dt.total_seconds(), pos_data[c]/10) for c in ["X","Y","Z"]])
                        speeds[i].append(np.interp(sts[i], car_data["Time"].dt.total_seconds(), car_data["Speed"]))

                avg_coords[sess_type] = np.mean(np.array(coords), axis=1)
                avg_speeds[sess_type] = np.mean(np.array(speeds), axis=1)

            avg_dist = np.sqrt(( avg_coords['R'] - avg_coords['Q'] )[:,0]**2 + ( avg_coords['R'] - avg_coords['Q'] )[:,1]**2 + ( avg_coords['R'] - avg_coords['Q'] )[:,2]**2 )
            print(f"{year}|{round_no}|{circuit_name}|{'|'.join(map(str, avg_dist))}|{'|'.join(map(str, avg_speeds['Q']))}")
        except Exception as e:
            # pass
            print(f"{year}|{round_no}|Error: {e}")