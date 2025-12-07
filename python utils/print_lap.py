import fastf1
import my_f1_utils # cache

session = fastf1.get_session(2025, 24, "Q")
session.load()
laps = session.laps.pick_drivers("NOR")
# lap = laps[laps["LapNumber"] == 5].iloc[0]
# lap = laps[laps["LapNumber"] == 40]

# session = fastf1.get_session(2024, 22, "Q")
# session.load()
# laps = session.laps.split_qualifying_sessions()[2]
# laps = laps.pick_drivers("RUS")
lap = laps.pick_fastest()

sector_times = [
    lap["Sector1Time"].total_seconds(),
    lap["Sector2Time"].total_seconds(),
    lap["Sector3Time"].total_seconds(),
]
print(f"{'|'.join(map(str, sector_times))}")

car_data = lap.get_car_data(pad=1, pad_side='both')
for row in car_data.itertuples():
    print(f"{row.Time.total_seconds()}|{row.Speed}|{row.Throttle}|{row.Brake}")

pos_data = lap.get_pos_data(pad=1, pad_side='both')
for row in pos_data.itertuples():
    print(f"{row.Time.total_seconds()}|{row.X/10}|{row.Y/10}|{row.Z/10}")