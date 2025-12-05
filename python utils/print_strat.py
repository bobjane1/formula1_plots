import fastf1, my_f1_utils

year = 2022
roundno = 22
session_type = "R"
session = fastf1.get_session(year, roundno, session_type)
session.load()
laps = session.laps
drivers = laps['Driver'].unique()
for driver in drivers:
    driver_laps = laps.pick_drivers(driver)
    stints = driver_laps[["Driver", "Stint", "Compound", "LapNumber"]]
    stints = stints.groupby(["Driver", "Stint", "Compound"])
    stints = stints.count().reset_index()
    stints = stints.rename(columns={"LapNumber": "StintLength"})
    stints = stints.sort_values(by=["Stint"])
    compound_str = "-".join(stints["Compound"].tolist())
    str_lengths = stints["StintLength"].tolist()
    cum_str_lengths = [sum(str_lengths[:i+1]) for i in range(len(str_lengths))]
    cum_str_lengths = "-".join([str(x) for x in cum_str_lengths])
    print(f"{driver}|{compound_str}|[{cum_str_lengths}]")