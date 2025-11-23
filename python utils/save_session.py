import fastf1, fastf1.plotting
import my_f1_utils # set the cache
from pathlib import Path
import pandas as pd

def save_driver_info(races):
    fn = "github/driver_info.csv"
    path = Path(fn)

    for year, round_no, session_type in races:
        try:
            print(f"Processing driver info for {year} {round_no} {session_type}")
            session = fastf1.get_session(year, round_no, session_type)
            session.load()
            assert len(session.results) > 0, "No results data found"

            drivers = session.results["Abbreviation"].to_list()
            colors = []
            for driver in drivers:
                try:
                    driver_row = session.results[session.results["Abbreviation"] == driver].iloc[0]
                    color = fastf1.plotting.get_driver_style(identifier=driver, style=['color'], session=session)["color"]
                    colors.append({"year": year, "round_no": round_no, "session_type": session_type, "driver": driver, "color": color, "TeamName": driver_row["TeamName"], "FullName": driver_row["FullName"], "HeadshotUrl": driver_row["HeadshotUrl"]})
                except Exception as e:
                    print(f"  Skipping driver {driver}: {e}")
                    continue
            df = pd.DataFrame(colors)

            if path.exists() and path.stat().st_size > 0:
                header_order = pd.read_csv(fn, nrows=0).columns.tolist()  # existing CSV column order
                df = df.reindex(columns=header_order)                      # reorder/align to file
                df.to_csv(fn, mode="a", index=False, header=False)         # append rows
            else:
                df.to_csv(fn, index=False)                                 # new/empty file: write header once
                        
        except Exception as e:
            print(f"Skipping {year} R{round_no}: {e}")
            continue

def process_round_laps(year, round_no, session_type):
    rows = []
    print(f"Processing round laps for {year} {session_type}{round_no}")
    session = fastf1.get_session(year, round_no, session_type)
    session.load()
    laps = session.laps
    circuit_name = session.session_info["Meeting"]["Circuit"]["ShortName"]
    meeting_name = session.session_info["Meeting"]["Name"]
    session_results = session.results
    sched_laps = session.total_laps

    for driver in laps["Driver"].unique():
        print(driver,end =": ")
        driver_results = session_results[session_results["Abbreviation"] == driver]
        assert len(driver_results) == 1
        finish_position = driver_results["Position"].values[0]
        grid_position = driver_results["GridPosition"].values[0]
        finish_status = driver_results["Status"].values[0]
        print(finish_status)
        driver_laps = laps.pick_drivers(driver)
        for _,lap in driver_laps.iterlaps():
            print(int(lap["LapNumber"]), end="|")

            # calc the min distance to car ahead
            car_data = lap.get_car_data().add_distance().add_driver_ahead()
            distances = car_data["DistanceToDriverAhead"].dropna()
            min_dist = 10000
            if len(distances) > 0:
                min_dist = distances.min()

            row = lap.to_dict()
            row = {f"lap_{k}": v for k, v in row.items()}
            weather_data = lap.get_weather_data().to_dict()
            weather_data = {f"weather_{k}": v for k, v in weather_data.items()}
            row.update(weather_data)

            row["year"] = year
            row["round_no"] = round_no
            row["session_type"] = session_type
            row["meeting_name"] = meeting_name
            row["circuit_name"] = circuit_name
            row["sched_laps"] = sched_laps
            row["grid_position"] = grid_position
            row["finish_position"] = finish_position
            row["finish_status"] = finish_status
            row["min_dist"] = min_dist
            row["lap_length"] = car_data["Distance"].iloc[-1]

            td_cols = [
                "lap_Time",
                "lap_LapTime",
                "lap_Sector1Time",
                "lap_Sector2Time",
                "lap_Sector3Time",
                "lap_Sector1SessionTime",
                "lap_Sector2SessionTime",
                "lap_Sector3SessionTime",
                "lap_LapStartTime",
                "lap_PitOutTime",
                "lap_PitInTime",
                "weather_Time",
            ]
            for col in td_cols:
                if row[col] is not None:
                    row[col] = row[col].total_seconds()
            rows.append(row)            
        print()
    print("\n")
        
    df = pd.DataFrame(rows)
    return df

def save_to_all_df(races):
    fn = "github/all_df.csv"
    
    for year, round_no, session_type in races:    
        try:
            df = process_round_laps(year, round_no, session_type)
            path = Path(fn)
            if path.exists() and path.stat().st_size > 0:
                header_order = pd.read_csv(fn, nrows=0).columns.tolist()  # existing CSV column order
                df = df.reindex(columns=header_order)                      # reorder/align to file
                df.to_csv(fn, mode="a", index=False, header=False)         # append rows
            else:
                df.to_csv(fn, index=False)                                 # new/empty file: write header once
        except Exception as e:
            print(f"Skipping {year} R{round_no}: {e}")
            continue

my_f1_utils.setup_cache(offline=False)
races = [(2025,22,"R")]
do_driver_info = True
do_all_df = True

if do_driver_info:
    save_driver_info(races)

if do_all_df:
    save_to_all_df(races)