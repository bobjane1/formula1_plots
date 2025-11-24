import logging, fastf1, numpy as np, pandas as pd, math, os
import statsmodels.formula.api as smf

def setup_cache(offline=True):
    fastf1.Cache.enable_cache("fastf1_cache")
    if offline:
        fastf1.logger.set_log_level(logging.ERROR)
        fastf1.Cache.offline_mode(True)

setup_cache()

def load_df():
    fn = "github/all_df.csv"
    # if on windows the fn is different
    if os.name == 'nt':
        fn = "main/all_df.csv"
    df = pd.read_csv(fn, dtype={"lap_TrackStatus": str}, low_memory=False)
    return df

def get_uniq(df,col):
    vals = df[col].unique()
    if len(vals) == 0: return None
    assert len(vals) == 1
    return vals[0]

def regression_ols(df, formula):
    res_ols = smf.ols(formula, data=df).fit()
    coefficients = res_ols.params
    predictions = res_ols.predict(df) 
    rmse = float(np.sqrt(np.mean((df[formula.split('~')[0].strip()] - predictions) ** 2)))
    r_squared = res_ols.rsquared
    return coefficients, math.sqrt(r_squared), rmse, 

def regression_2vars(xs,ys):
    corr = xs.corr(ys)
    beta = float(np.std(ys)) / float(np.std(xs)) * corr
    alpha = ys.mean() - beta * xs.mean()
    rmse = float(np.sqrt(np.mean((ys - (alpha + beta * xs)) ** 2)))
    return alpha, beta, corr, rmse

def remove_nas(laps,cols=["lap_LapTime", "lap_TyreLife"]):
    for col in cols:
        laps = laps[~laps[col].isna()]
    return laps

def filter_safety_car(df):
    df = df[df['lap_TrackStatus'].str.contains('[4567]') == False]
    return df

def filter_rainfall(df):
    df = df[~df["weather_Rainfall"]]
    return df

def quick_laps(df,sched_laps=10000,status=None,dist_thresh=10, start_lap_num=2):
    df = df[(df["lap_LapNumber"] < sched_laps - 5) & (df["lap_LapNumber"] >= start_lap_num)]
    if status is not None: df = df[df["finish_status"].isin(status)]
    df = filter_safety_car(df)
    df = filter_rainfall(df)
    df = df[df["lap_PitInTime"].isna()]
    df = df[df["lap_PitOutTime"].isna()]
    df = df[df["min_dist"] > dist_thresh]
    return df

def tyre_strategy_summary(df):
    rows = []
    for (driver,), driver_df in df.groupby(["lap_Driver"]):
        row = {
            "driver": driver,
            "laps_completed": driver_df["lap_LapNumber"].max(),
            "grid_position": get_uniq(driver_df, "grid_position"),
            "finish_position": get_uniq(driver_df, "finish_position"),
            "finish_status": get_uniq(driver_df, "finish_status"),
        }

        for (stint,), stint_df in driver_df.groupby(["lap_Stint"]):
            stint = int(stint)
            row[f"stint{stint+1}_compound"] = get_uniq(stint_df, "lap_Compound")
            row[f"stint{stint+1}_start_lap"] = int(stint_df["lap_LapNumber"].min())
            row[f"stint{stint+1}_end_lap"] = int(stint_df["lap_LapNumber"].max())
            row[f"stint{stint+1}_start_tyre_life"] = int(stint_df["lap_TyreLife"].min())
        rows.append(row)
    
    ans = pd.DataFrame(rows)
    return ans

def stint_regressions(meeting_df):
    stints = []
    for (driver,session_type), session_df in meeting_df.groupby(["lap_Driver","session_type"]):
        sched_laps = get_uniq(session_df, "sched_laps")
        for (stint,), stint_df in session_df.groupby(["lap_Stint"]):
            stint = int(stint)
            min_lap_num = stint_df["lap_LapNumber"].min() # remove the lowest lap # of a stint (cold tyres)
            stint_df = stint_df[stint_df["lap_LapNumber"] != min_lap_num]
            stint_quick_df = quick_laps(stint_df,sched_laps,["Finished","Lapped","Retired"])
            if len(stint_quick_df) < 7: continue # too short to analyze

            compound = get_uniq(stint_df, "lap_Compound")
            stint_start_lap = int(stint_df["lap_LapNumber"].min())
            stint_start_tyre_life = int(stint_df["lap_TyreLife"].min())
            alpha, beta, corr, rmse = regression_2vars(stint_quick_df["lap_TyreLife"], stint_quick_df["lap_LapTime"])
            
            stints.append({
                "driver": driver,
                "compound": compound,
                "alpha": alpha, 
                "beta": beta, 
                "npoints": len(stint_quick_df),
                "corr": corr,
                "rmse": rmse,
                "session_type": session_type,
                "stint": stint,
                "start_tyre_life": stint_start_tyre_life,
                "laps_to_go": sched_laps - stint_start_lap,
            })

    # for driver, driver_stints_by_compound in ans.items():
        # for compound, stints in driver_stints_by_compound.items():
    #         if len(stints) < 2: continue
            # for stint in stints:
                # print(f"{driver}|{stint['session_type']}|{stint['stint']}|{stint['start_tyre_life']}|{stint['laps_to_go']}|{compound}|{stint['alpha']}|{stint['beta']}")

    return stints

def adjust_stint_regression(alpha, beta, start_tyre_life, laps_to_go, fuel_effect):
    adjusted_alpha = alpha + (start_tyre_life + laps_to_go) * fuel_effect
    adjusted_beta = beta - fuel_effect
    return adjusted_alpha, adjusted_beta

def load_driver_colors():
    df = pd.read_csv("github/driver_info.csv")
    return df.set_index(["year", "round_no", "driver"])["color"].to_dict()