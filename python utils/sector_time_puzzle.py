import fastf1, logging, numpy as np, matplotlib.pyplot as plt
fastf1.Cache.enable_cache("fastf1_cache")
fastf1.logger.set_log_level(logging.ERROR)
fastf1.Cache.offline_mode(True)

avgs = {}
for i, sess_type in enumerate(["Q","R"]):
    session = fastf1.get_session(2024, "Las Vegas", sess_type)
    session.load()
    if sess_type == "R":
        laps = session.laps
        lap = laps[(laps["Driver"]=="RUS") & (laps["LapNumber"]==18)].iloc[0] # random lap to make the track
        plt.plot(lap.get_pos_data()["X"]/10, lap.get_pos_data()["Y"]/10, label="RUS Q18", alpha=0.5, color="gray")
    coords = [[],[],[]]    
    for _,lap in session.laps.iterrows():
        if np.isnan(lap["Sector1Time"].total_seconds()) or np.isnan(lap["Sector2Time"].total_seconds()) or np.isnan(lap["Sector3Time"].total_seconds()): 
            continue
        pos_data = lap.get_pos_data(pad=2, pad_side='both')
        interp_t = lambda t: [np.interp(t, pos_data["Time"].dt.total_seconds(), pos_data[c]/10) for c in ["X","Y"]]
        coords[0].append(interp_t(lap["Sector1Time"].total_seconds()))
        coords[1].append(interp_t(lap["Sector1Time"].total_seconds()+lap["Sector2Time"].total_seconds()))
        coords[2].append(interp_t(lap["Sector1Time"].total_seconds()+lap["Sector2Time"].total_seconds()+lap["Sector3Time"].total_seconds()))
    
    for idx in range(3):
        plt.scatter(*zip(*coords[idx]), label=f"{sess_type} S{idx+1}")

plt.legend()
plt.ticklabel_format(style='plain', axis='both')
plt.show()