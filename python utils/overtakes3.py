import my_f1_utils, numpy as np

def calc_gaps(df):
    sched_laps = my_f1_utils.get_uniq(df, "sched_laps")
    df = df[~df["lap_Position"].isna()]     # filter our laps with position na, those are dnfs
    gaps_by_lap = []
    for lap in range(1, sched_laps + 1):
        lap_df = df[df["lap_LapNumber"] == lap].sort_values(by=["lap_Position"])
        drivers = list(lap_df["lap_Driver"].values)
        lap_df = lap_df.set_index("lap_Driver")
        first_driver_time = lap_df.at[drivers[0], "lap_Time"]
        finish_times = [[drv, lap_df.at[drv, "lap_Time"] - first_driver_time] for drv in drivers]
        gaps_by_lap.append(finish_times)
    return gaps_by_lap

def find_consecutive_pairs(gaps_by_lap):
    ans = []
    active_pairs = {}  # key: (driver_front, driver_behind), value: {from_lap, gaps, last_positions}

    for lap_num, lap_data in enumerate(gaps_by_lap, start=1):
        current_pairs = set() # lap_data is list of [driver, gap_to_leader]

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
                reason = None
                if driver_front in driver_positions and driver_behind in driver_positions:
                    current_pos_front = driver_positions[driver_front]
                    current_pos_behind = driver_positions[driver_behind]

                    if current_pos_behind < current_pos_front:
                        # Driver behind overtook driver front
                        positions_ahead = current_pos_front - current_pos_behind
                        reason = {"type": "overtake", "pos_ahead": positions_ahead}
                    else:
                        # They're further apart, determine who changed position
                        pos_change_front = current_pos_front - last_pos_front  # negative = gained positions
                        pos_change_behind = current_pos_behind - last_pos_behind  # positive = lost positions
                        reason = {"type": "gap_grew", "pos_change_front": pos_change_front, "pos_change_behind": pos_change_behind}
                elif driver_front not in driver_positions:
                    reason = {"type": "front out of race"}
                elif driver_behind not in driver_positions:
                    reason = {"type": "behind out of race"}

                pair_data["end_reason"] = reason
                pair_data["end_lap"] = lap_num
                del pair_data["last_positions"]  # Remove internal tracking data

                ans.append(pair_data)
                pairs_to_remove.append(pair_key)

        for pair_key in pairs_to_remove:
            del active_pairs[pair_key]

    # Handle pairs that lasted until the end of the race
    for pair_data in active_pairs.values():
        pair_data["end_reason"] = {"type": "race_end"}
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
    
    gaps_by_lap = calc_gaps(df) # for each lap, list of [driver, gap to leader]

    consecutive_pairs = find_consecutive_pairs(gaps_by_lap)

    for pair in consecutive_pairs:
        if len(pair["gaps"]) < 3: continue # cashing for 3 laps
        if pair["gaps"][-1] > 3: continue # got close enough
        if pair["gaps"][0] - pair["gaps"][-1] < 2: continue # closed the gap
        if team_map[pair["driver_front"]] == team_map[pair["driver_behind"]]: continue # same team

        # if pair["end_reason"]["type"] == "gap_grew": continue
        print(f"{pair['driver_behind']} chasing {pair['driver_front']} for {len(pair['gaps'])} laps (lap {pair['from_lap']}-{pair['end_lap']})")
        print(f"  Gaps: {[f'{g:.3f}' for g in pair['gaps']]}")
        print(f"  End reason: {pair['end_reason']}")
        print()

main()