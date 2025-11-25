import numpy as np
import fastf1
import fastf1.api
import my_f1_utils

years = list(range(2025,2017,-1))
rounds = list(range(26,0,-1))
session_types = ["R","S"]

years = [2025]
rounds = [22]
session_types = ["R"]

for year in years:
    for round_no in rounds:
        for session_type in session_types:
            fn = f"github/timing_data/{year}_{round_no}_{session_type}.csv"
            try:
                session = fastf1.get_session(year, round_no, session_type)
                session.load()
                timing_data = fastf1.api.timing_data(session.api_path)[1]
                timing_data = timing_data[(timing_data["Driver"] == "4") & (timing_data["Position"] == 2)]
                print(timing_data.iloc[0]["Time"])
                print(timing_data.iloc[0]["Time"].total_seconds())
                # timing_data["Time"] = timing_data["Time"].dt.total_seconds()
                # timing_data.to_csv(fn, index=False) 
                print(f"Saved {fn}")
            except Exception as e:
                print(f"Skipped {fn}: {e}")
                continue
