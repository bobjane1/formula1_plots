import fastf1, my_f1_utils
import pandas as pd

# my_f1_utils.setup_cache(offline=False)
year = 2025
roundno = 24
session_type = "FP3"

session = fastf1.get_session(year, roundno, session_type)
session.load()
laps = session.laps
drivers = laps['Driver'].unique()
for driver in drivers:
    # if driver not in ["VER", "NOR", "PIA", "HAM", "LEC", "SAI", "RUS", "ANT"]: continue
    # if driver != "BOR": continue
    driver_laps = laps.pick_drivers(driver)
    for li, lap in driver_laps.iterlaps():
        if lap["Compound"] != "HARD": continue
        temp = lap.get_weather_data()["TrackTemp"]    
        if pd.isna(lap["LapTime"]) or lap["LapTime"].total_seconds() > 95: continue
        print(f"{driver}|{temp}|{lap['LapNumber']}|{lap['Compound']}|{lap['TyreLife']}|{lap['Sector1Time'].total_seconds()}|{lap['Sector2Time'].total_seconds()}|{lap['Sector3Time'].total_seconds()}|{lap['LapTime'].total_seconds()}")
    
    # stints = driver_laps[["Driver", "Stint", "Compound", "LapNumber"]]
    # stints = stints.groupby(["Driver", "Stint", "Compound"])
    # stints = stints.count().reset_index()
    # stints = stints.rename(columns={"LapNumber": "StintLength"})
    # stints = stints.sort_values(by=["Stint"])
    # compound_str = "-".join(stints["Compound"].tolist())
    # str_lengths = stints["StintLength"].tolist()
    # cum_str_lengths = [sum(str_lengths[:i+1]) for i in range(len(str_lengths))]
    # cum_str_lengths = "-".join([str(x) for x in cum_str_lengths])
    # print(f"{driver}|{compound_str}|[{cum_str_lengths}]")