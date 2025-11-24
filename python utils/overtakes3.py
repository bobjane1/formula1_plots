import my_f1_utils, numpy as np, pandas as pd

def load_lap_data(df):
    sched_laps = my_f1_utils.get_uniq(df, "sched_laps")
    df = df[~df["lap_Position"].isna()]     # filter our laps with position na, those are dnfs
    gaps_by_lap = []
    pit_stops_by_lap = []  # track pit stops for each lap
    safety_car_by_lap = []  # track safety car laps for each driver
    for lap in range(1, sched_laps + 1):
        lap_df = df[df["lap_LapNumber"] == lap].sort_values(by=["lap_Position"])
        drivers = list(lap_df["lap_Driver"].values)
        lap_df = lap_df.set_index("lap_Driver")
        p1_lap_time = lap_df.at[drivers[0], "lap_Time"]
        gaps_to_p1 = [[drv, lap_df.at[drv, "lap_Time"] - p1_lap_time] for drv in drivers]
        gaps_by_lap.append(gaps_to_p1)

        # Track which drivers pitted on this lap
        pitted_drivers = set()
        for drv in drivers:
            if not pd.isna(lap_df.at[drv, "lap_PitInTime"]) or not pd.isna(lap_df.at[drv, "lap_PitOutTime"]):
                pitted_drivers.add(drv)
        pit_stops_by_lap.append(pitted_drivers)

        # Track which drivers had safety car on this lap (track status 4, 5, 6, or 7)
        safety_car_drivers = set()
        for drv in drivers:
            track_status = str(lap_df.at[drv, "lap_TrackStatus"])
            if any(status in track_status for status in ['4', '5', '6', '7']):
                safety_car_drivers.add(drv)
        safety_car_by_lap.append(safety_car_drivers)

    return gaps_by_lap, pit_stops_by_lap, safety_car_by_lap

def calc_battles(gaps_by_lap, pit_stops_by_lap, safety_car_by_lap):
    ans = []
    active_pairs = {}  # key: (driver_front, driver_behind), value: {from_lap, gaps, last_positions}

    def collect_post_overtake_gaps(start_lap, overtaker, overtaken):
        """Collect gaps while overtaker stays directly ahead of overtaken."""
        post_gaps = []
        for lap_idx in range(start_lap - 1, len(gaps_by_lap)):
            lap_entries = gaps_by_lap[lap_idx]
            driver_order = [entry[0] for entry in lap_entries]

            if overtaker not in driver_order or overtaken not in driver_order:
                break

            overtaker_pos = driver_order.index(overtaker)
            overtaken_pos = driver_order.index(overtaken)

            # Stop once they are no longer consecutive with overtaker ahead
            if overtaker_pos + 1 != overtaken_pos:
                break

            gap_between = lap_entries[overtaken_pos][1] - lap_entries[overtaker_pos][1]
            gap_to_front = None
            if overtaker_pos > 0:
                gap_to_front = lap_entries[overtaker_pos][1] - lap_entries[overtaker_pos - 1][1]

            post_gaps.append([float(gap_between), float(gap_to_front)])

        return post_gaps

    for lap_num, lap_data in enumerate(gaps_by_lap, start=1):
        current_pairs = set() # lap_data is list of [driver, gap_to_leader]
        pitted_drivers = pit_stops_by_lap[lap_num - 1]  # get pit stops for this lap
        safety_car_drivers = safety_car_by_lap[lap_num - 1]  # get safety car status for this lap

        # Create position lookup for this lap
        driver_positions = {lap_data[i][0]: i + 1 for i in range(len(lap_data))}

        for i in range(len(lap_data) - 1):
            gap_between = lap_data[i + 1][1] - lap_data[i][1]
            pair_key = (lap_data[i][0], lap_data[i + 1][0])
            current_pairs.add(pair_key)

            if pair_key in active_pairs:
                # Continue existing sequence
                active_pairs[pair_key]["gaps"].append(gap_between)
                active_pairs[pair_key]["last_positions"] = (i + 1, i + 2)
            else:
                # Start new sequence
                active_pairs[pair_key] = {
                    "driver_front": pair_key[0],
                    "driver_behind": pair_key[1],
                    "from_lap": lap_num,
                    "gaps": [gap_between],
                    "last_positions": (i + 1, i + 2)  # (pos_front, pos_behind)
                }

        # Check which pairs are no longer consecutive and finalize them
        pairs_to_remove = []
        for pair_key in active_pairs:
            if pair_key not in current_pairs:
                pair_data = active_pairs[pair_key]
                driver_front, driver_behind = pair_key
                last_pos_front, last_pos_behind = pair_data["last_positions"]

                # Determine why they're no longer consecutive
                event = None

                # Check if either driver had safety car on this lap
                if driver_front in safety_car_drivers or driver_behind in safety_car_drivers:
                    event = {"type": "safety_car"}
                # Check if either driver pitted on this lap
                elif driver_front in pitted_drivers:
                    event = {"type": "pit_stop_front"}
                elif driver_behind in pitted_drivers:
                    event = {"type": "pit_stop_behind"}
                elif driver_front in driver_positions and driver_behind in driver_positions:
                    current_pos_front = driver_positions[driver_front]
                    current_pos_behind = driver_positions[driver_behind]

                    if current_pos_behind < current_pos_front:
                        # Driver behind overtook driver front
                        positions_ahead = current_pos_front - current_pos_behind
                        event = {
                            "type": "overtake",
                            "pos_ahead": positions_ahead,
                            "post_overtake_gaps": collect_post_overtake_gaps(
                                lap_num, driver_behind, driver_front
                            )
                        }
                    else:
                        # They're further apart, determine who changed position
                        pos_change_front = current_pos_front - last_pos_front  # negative = gained positions
                        pos_change_behind = current_pos_behind - last_pos_behind  # positive = lost positions
                        event = {"type": "gap_grew", "pos_change_front": pos_change_front, "pos_change_behind": pos_change_behind}
                elif driver_front not in driver_positions:
                    event = {"type": "front out of race"}
                elif driver_behind not in driver_positions:
                    event = {"type": "behind out of race"}

                pair_data["event"] = event
                pair_data["end_lap"] = lap_num
                del pair_data["last_positions"]  # Remove internal tracking data

                ans.append(pair_data)
                pairs_to_remove.append(pair_key)

        for pair_key in pairs_to_remove:
            del active_pairs[pair_key]

    # Handle pairs that lasted until the end of the race
    for pair_data in active_pairs.values():
        pair_data["event"] = {"type": "race_end"}
        pair_data["end_lap"] = len(gaps_by_lap)
        del pair_data["last_positions"]
        ans.append(pair_data)

    return ans

def main():
    all_df = my_f1_utils.load_df()
    year, round_no, session_type = 2025, 21, "R"
    df = all_df[(all_df["year"] == year) & (all_df["round_no"] == round_no) & (all_df["session_type"] == session_type)]

    # get the unique lap_Team for each lap_Driver
    team_map = {}
    for drv, drv_df in df.groupby("lap_Driver"):
        team_map[drv] = my_f1_utils.get_uniq(drv_df, "lap_Team")
    
    gaps_by_lap, pit_stops_by_lap, safety_car_by_lap = load_lap_data(df) # for each lap, list of [driver, gap to leader]

    battles = calc_battles(gaps_by_lap, pit_stops_by_lap, safety_car_by_lap)

    for battle in battles:
        if len(battle["gaps"]) < 3: continue # cashing for 3 laps
        if battle["gaps"][-1] > 3: continue # got close enough
        if battle["gaps"][0] - battle["gaps"][-1] < 2: continue # closed the gap
        if team_map[battle["driver_front"]] == team_map[battle["driver_behind"]]: continue # same team

        print(f"{battle['driver_behind']}|{battle['driver_front']}|{len(battle['gaps'])}|{battle['from_lap']}|{battle['end_lap']}|{battle['event']['type']}",end="|")
        print("|".join(map(str,battle['gaps'])),end="|")

        if battle['event']['type'] == 'overtake':
            print("|".join(map(str, [x[0]*(-1) for x in battle['event']['post_overtake_gaps']])))
        else:
            print()

main()
