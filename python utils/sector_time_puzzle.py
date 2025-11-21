import fastf1, logging, numpy as np, matplotlib.pyplot as plt
fastf1.Cache.enable_cache("fastf1_cache")
fastf1.logger.set_log_level(logging.ERROR)
fastf1.Cache.offline_mode(True)
for i, sess_type in enumerate(["Q","R"]):
    session = fastf1.get_session(2024, "Las Vegas", sess_type)
    session.load()
    if sess_type == "R":
        laps = session.laps
        lap = laps[(laps["Driver"]=="RUS") & (laps["LapNumber"]==18)].iloc[0] # random lap to make the track
        plt.plot(lap.get_pos_data()["X"]/10, lap.get_pos_data()["Y"]/10, label="RUS Q18", alpha=0.5, color="gray")
    coords1 = []
    coords2 = []
    for _,lap in session.laps.iterrows():
        s1_t = lap["Sector1Time"].total_seconds()
        if np.isnan(s1_t): continue
        pos_data = lap.get_pos_data()
        coords1.append([np.interp(lap["Sector1Time"].total_seconds(), pos_data["Time"].dt.total_seconds(), pos_data[c]/10) for c in ["X","Y"]])
        coords2.append([np.interp(lap["Sector1Time"].total_seconds()+lap["Sector2Time"].total_seconds(), pos_data["Time"].dt.total_seconds(), pos_data[c]/10) for c in ["X","Y"]])
    plt.scatter(*zip(*coords1), label=sess_type+" S1")
    plt.scatter(*zip(*coords2), label=sess_type+" S2")
plt.legend()
plt.ticklabel_format(style='plain', axis='both')
plt.show()