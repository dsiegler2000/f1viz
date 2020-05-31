import argparse
import os
import shutil
from collections import defaultdict

import flag
import pandas as pd
import numpy as np
from utils import millis_to_str, NATIONALITY_TO_FLAG

"""
Does the bulk of the pre-calculation
"""

parser = argparse.ArgumentParser()
parser.add_argument("--all", nargs="?", const=True, default=False, help="Activate all modes (default).")
parser.add_argument("--custom", nargs="?", const=True, default=False, help="Custom mode (use flags to specify what "
                                                                           "should be run).")
parser.add_argument("--load_from_data", nargs="?", const=True, default=False, help="Load directly from the data folder "
                                                                                   "rather  than temp_data.")
parser.add_argument("--enable_weather_sc", nargs="?", const=True, default=False, help="Weather safety car.")
parser.add_argument("--enable_ratings", nargs="?", const=True, default=False, help="Ratings.")
parser.add_argument("--enable_runtime", nargs="?", const=True, default=False, help="Runtime.")
parser.add_argument("--enable_round_num_name", nargs="?", const=True, default=False, help="Round number and name.")
parser.add_argument("--enable_imgs", nargs="?", const=True, default=False, help="Image.")
parser.add_argument("--enable_race_mlt", nargs="?", const=True, default=False, help="Race mean lap time.")
parser.add_argument("--enable_fastest_lap", nargs="?", const=True, default=False, help="Fastest lap info.")

args = parser.parse_args()
if not args.custom:
    args.enable_weather_sc = True
    args.enable_ratings = True
    args.enable_runtime = True
    args.enable_round_num_name = True
    args.enable_imgs = True
    args.enable_race_mlt = True
    args.enable_fastest_lap = True
else:
    args.load_from_data = True

# Read in all of the data
if args.load_from_data:
    data_dir = os.path.join("data")
    circuits = "circuits_augmented.csv"
    constructor_results = "constructor_results_augmented.csv"
    constructor_standings = "constructor_standings_augmented.csv"
    constructors = "constructors_augmented.csv"
    driver_standings = "driver_standings_augmented.csv"
    drivers = "drivers_augmented.csv"
    lap_times = "lap_times_augmented.csv"
    pit_stops = "pit_stops_augmented.csv"
    qualifying = "qualifying_augmented.csv"
    races = "races_augmented.csv"
    results = "results_augmented.csv"
    seasons = "seasons_augmented.csv"
    status = "status_augmented.csv"
    circuits = pd.read_csv(os.path.join(data_dir, circuits), index_col=0)
    constructor_results = pd.read_csv(os.path.join(data_dir, constructor_results), index_col=0)
    constructor_standings = pd.read_csv(os.path.join(data_dir, constructor_standings), index_col=0)
    constructors = pd.read_csv(os.path.join(data_dir, constructors), index_col=0)
    driver_standings = pd.read_csv(os.path.join(data_dir, driver_standings), index_col=0)
    drivers = pd.read_csv(os.path.join(data_dir, drivers), index_col=0)
    lap_times = pd.read_csv(os.path.join(data_dir, lap_times), index_col=0)
    pit_stops = pd.read_csv(os.path.join(data_dir, pit_stops), index_col=0)
    qualifying = pd.read_csv(os.path.join(data_dir, qualifying), index_col=0)
    races = pd.read_csv(os.path.join(data_dir, races), index_col=0)
    results = pd.read_csv(os.path.join(data_dir, results), index_col=0)
    seasons = pd.read_csv(os.path.join(data_dir, seasons), index_col=0)
    status = pd.read_csv(os.path.join(data_dir, status), index_col=0)
else:
    data_dir = os.path.join("temp_data", "cleaned")
    circuits = "circuits_cleaned.csv"
    constructor_results = "constructor_results_cleaned.csv"
    constructor_standings = "constructor_standings_cleaned.csv"
    constructors = "constructors_cleaned.csv"
    driver_standings = "driver_standings_cleaned.csv"
    drivers = "drivers_cleaned.csv"
    lap_times = "lap_times_cleaned.csv"
    pit_stops = "pit_stops_cleaned.csv"
    qualifying = "qualifying_cleaned.csv"
    races = "races_cleaned.csv"
    results = "results_cleaned.csv"
    seasons = "seasons_cleaned.csv"
    status = "status_cleaned.csv"
    circuits = pd.read_csv(os.path.join(data_dir, circuits), index_col=0).set_index("circuitId")
    constructor_results = pd.read_csv(os.path.join(data_dir, constructor_results), index_col=0)\
        .set_index("constructorResultsId")
    constructor_standings = pd.read_csv(os.path.join(data_dir, constructor_standings), index_col=0)\
        .set_index("constructorStandingsId")
    constructors = pd.read_csv(os.path.join(data_dir, constructors), index_col=0).set_index("constructorId")
    driver_standings = pd.read_csv(os.path.join(data_dir, driver_standings), index_col=0)\
        .set_index("driverStandingsId")
    drivers = pd.read_csv(os.path.join(data_dir, drivers), index_col=0).set_index("driverId")
    lap_times = pd.read_csv(os.path.join(data_dir, lap_times), index_col=0)
    pit_stops = pd.read_csv(os.path.join(data_dir, pit_stops), index_col=0)
    qualifying = pd.read_csv(os.path.join(data_dir, qualifying), index_col=0).set_index("qualifyId")
    races = pd.read_csv(os.path.join(data_dir, races), index_col=0)
    results = pd.read_csv(os.path.join(data_dir, results), index_col=0).set_index("resultId")
    seasons = pd.read_csv(os.path.join(data_dir, seasons), index_col=0).set_index("year")
    status = pd.read_csv(os.path.join(data_dir, status), index_col=0).set_index("statusId")


# Define some util methods
def get_driver_name(did):
    driver = drivers.loc[did]
    name = driver["forename"] + " " + driver["surname"]
    nat = driver["nationality"].lower()
    if nat in NATIONALITY_TO_FLAG.index:
        flag_t = NATIONALITY_TO_FLAG.loc[nat, "flag"]
        name = flag.flagize(f":{flag_t}: " + name)
    return name


def get_race_name(rid):
    race = races.loc[rid]
    circuit = circuits.loc[race["circuitId"]]
    nat = circuit["country"].lower()
    name = circuit["country"]
    flag_t = NATIONALITY_TO_FLAG.loc[nat, "flag"]
    name = flag.flagize(f":{flag_t}: " + name)
    return name


def get_constructor_name(cid):
    try:
        constructor = constructors.loc[cid]
        name = constructor["name"]
        nat = constructor["nationality"].lower()
        if nat in NATIONALITY_TO_FLAG.index:
            flag_t = NATIONALITY_TO_FLAG.loc[nat, "flag"]
            name = flag.flagize(f":{flag_t}: " + name)
        return name
    except KeyError:
        return "UNKNOWN"

# ======================================================================================================================
# Weather and safety car info
# ======================================================================================================================
if args.enable_weather_sc:
    weather_sc = pd.read_csv("data/static_data/Weather_SafetyCar.csv")
    weather_sc = weather_sc.rename(columns={"SC Laps": "SCLaps"})
    weather_sc["SCLaps"] = weather_sc["SCLaps"].fillna(0.0)

    weather_sc["raceId"] = None
    weather_sc = weather_sc[["raceId", "year", "name", "weather", "SCLaps"]]
    weather_sc["weather"] = weather_sc["weather"].str.lower()
    weather_sc["SCLaps"] = weather_sc["SCLaps"].astype(int)

    races["weather"] = None
    races["SCLaps"] = None

    for index, row in weather_sc.iterrows():
        races_that_year = races[races["year"] == row["year"]]
        name = row["name"].lower().split(" ")[0]
        if name == "azerbaijan" and row["year"] == 2016:  # Quick special case
            name = "european"
        name_p = races_that_year["name"].str.lower().str.split(" ").str[0] == name
        race = races_that_year[name_p]

        # Add circuitId to original dataset
        weather_sc.at[index, "raceId"] = race.index.values[0]

        # If the SC data is null, they really mean 0
        if np.isnan(row["SCLaps"]):
            sc_laps = 0
        else:
            sc_laps = row["SCLaps"]

        # Add weather and SCLaps to the races dataset
        races.loc[race.index, "weather"] = row["weather"]
        races.loc[race.index, "SCLaps"] = sc_laps

    weather_sc["raceId"] = weather_sc["raceId"].astype(int)
    races["SCLaps"] = races["SCLaps"].astype(float)

# ======================================================================================================================
# Fan ratings
# ======================================================================================================================
if args.enable_ratings:
    fan_ratings = pd.read_csv("data/static_data/fan_ratings.csv")
    fan_ratings = fan_ratings.rename(columns={  # remap column names
        "Y": "year",
        "R": "raceNumber",
        "GPNAME": "name",
        "P1": "p1",
        "P2": "p2",
        "P3": "p3",
        "RATING": "rating"
    })
    fan_ratings = fan_ratings.sort_values(by=["year", "raceNumber"])  # sort
    fan_ratings = fan_ratings.dropna()
    fan_ratings = fan_ratings[fan_ratings["year"] != 2018]  # 2018 data is incomplete
    fan_ratings = fan_ratings.loc[:, ~fan_ratings.columns.str.contains("^Unnamed")].drop(columns="index", errors="ignore")

    fan_ratings["raceId"] = None
    races["rating"] = None
    for index, row in fan_ratings.iterrows():
        races_that_year = races[races["year"] == row["year"]]
        name = row["name"].lower().split(" ")[0]
        if name == "azerbaijan" and row["year"] == 2016:  # Quick special case
            name = "european"
        name_p = races_that_year["name"].str.lower().str.split(" ").str[0] == name
        race = races_that_year[name_p]

        # Add circuitId to original dataset
        fan_ratings.at[index, "raceId"] = race.index.values[0]

        # Add rating to races
        races.at[race.index.values[0], "rating"] = row["rating"]

    races["rating"] = races["rating"].astype(float)
    fan_ratings["raceId"] = fan_ratings["raceId"].astype(int)
    fan_ratings = fan_ratings.set_index("raceId")


# ======================================================================================================================
# Race runtime
# ======================================================================================================================
if args.enable_runtime:
    def get_race_runtime(results, row):
        rid = row.name
        return np.min(results[results["raceId"] == rid]["milliseconds"].dropna())


    races["runtime"] = races.apply(lambda x: get_race_runtime(results, x), axis=1)


# ======================================================================================================================
# Per-circuit race rating
# ======================================================================================================================
if args.enable_ratings:
    ratings = races.groupby("circuitId").agg("mean")["rating"]
    circuits["avgRating"] = None
    for index, row in circuits.iterrows():
        cid = index
        if cid in ratings.index:
            circuits.at[cid, "avgRating"] = ratings.loc[cid]

# ======================================================================================================================
# Round number and name
# ======================================================================================================================
if args.enable_round_num_name:
    print("On round number and name")
    driver_standings["roundNum"] = driver_standings["raceId"].apply(lambda x: races.loc[x, "round"])

    driver_standings["roundName"] = driver_standings["raceId"].apply(lambda x: get_race_name(x))
    constructor_standings["roundNum"] = constructor_standings["raceId"].apply(lambda x: races.loc[x, "round"])
    constructor_standings["roundName"] = constructor_standings["raceId"].apply(lambda x: get_race_name(x))

# ======================================================================================================================
# Circuit image URL
# ======================================================================================================================
if args.enable_imgs:
    img_url = pd.read_csv("data/static_data/circuit_image_urls.csv").set_index("circuitId")
    circuits["imgUrl"] = img_url
    img_url = pd.read_csv("data/static_data/driver_image_urls.csv").set_index("driverId")
    drivers["imgUrl"] = img_url
    print(drivers["imgUrl"].isna().sum())

# ======================================================================================================================
# Constructor per-race mean lap time (stored in driver_results)
# ======================================================================================================================
if args.enable_race_mlt:
    print("On mean lap time")
    constructor_df = pd.DataFrame(columns=["raceId", "constructorId", "constructor_name", "mean_time", "rank"])
    i = 0
    counter = 0
    for race_id in races.index:
        counter += 1
        if counter % 100 == 0:
            print(f"{counter} / {races.shape[0]}")
        race_laps = lap_times[lap_times["raceId"] == race_id]
        race_results = results[results["raceId"] == race_id]
        constructor_laps = defaultdict(lambda: [])
        for driver_id in race_laps["driverId"].unique():
            cid = race_results[race_results["driverId"] == driver_id]["constructorId"]
            if cid.shape[0] > 0:
                cid = cid.values[0]
            else:
                continue
            constructor_laps[cid].extend(race_laps[race_laps["driverId"] == driver_id]["milliseconds"].values.tolist())

        for cid, times in constructor_laps.items():
            constructor_df = constructor_df.append({
                "raceId": race_id,
                "constructorId": cid,
                "constructor_name": get_constructor_name(cid),
                "mean_time": np.mean(times),
                "rank": 0.0
            }, ignore_index=True)
            constructor_laps[cid] = i
            i += 1
        race_source = constructor_df[constructor_df["raceId"] == race_id].drop(columns=["raceId"])
        race_source = race_source.set_index("constructorId")
        ranks = race_source["mean_time"].rank()
        for cid, idx in constructor_laps.items():
            constructor_df.at[idx, "rank"] = int(ranks.loc[cid])
    constructor_df.to_csv("data/constructor_mean_lap_times.csv", encoding="utf-8")

# ======================================================================================================================
# Fastest lap info
# ======================================================================================================================
if args.enable_fastest_lap:
    print("On fastest lap")
    fastest_lap_data = []
    i = 1
    for race_id in races.index:
        race_data = {
            "driver_id": [],
            "name": [],
            "constructor_id": [],
            "constructor_name": [],
            "fastest_lap_time_millis": [],
            "fastest_lap_time_str": [],
            "avg_lap_time_millis": [],
            "avg_lap_time_str": [],
        }
        race_results = results[results["raceId"] == race_id]
        race_laps = lap_times[lap_times["raceId"] == race_id]
        use_lap_data = race_results["rank"].isna().sum() == race_results.shape[0] or race_results["rank"].sum() < 0.1 \
                       or race_results["rank"].unique().shape[0] == 1
        if not use_lap_data:
            race_data["rank"] = []
        for idx, results_row in race_results.iterrows():
            driver_id = results_row["driverId"]
            race_data["driver_id"].append(driver_id)
            name = get_driver_name(driver_id)
            constructor_id = results_row["constructorId"]
            constructor_name = get_constructor_name(constructor_id)

            driver_laps = race_laps[race_laps["driverId"] == driver_id]
            avg_lap_time_millis = driver_laps["milliseconds"].mean()

            if use_lap_data:
                fastest_lap_millis = driver_laps["milliseconds"].min()
                fastest_lap_str = millis_to_str(fastest_lap_millis)

                race_data["fastest_lap_time_millis"].append(fastest_lap_millis)
                race_data["fastest_lap_time_str"].append(fastest_lap_str)
            else:
                rank = results_row["rank"]
                rank = 0 if np.isnan(rank) or np.isinf(rank) else int(rank)
                fastest_lap_millis = results_row["fastestLapTime"]
                fastest_lap_time_str = millis_to_str(fastest_lap_millis)

                race_data["rank"].append(rank)
                race_data["fastest_lap_time_millis"].append(fastest_lap_millis)
                race_data["fastest_lap_time_str"].append(fastest_lap_time_str)
            if np.isnan(avg_lap_time_millis) and results_row["laps"] > 0:
                if np.isnan(results_row["position"]):
                    avg_lap_time_millis = np.nan
                else:
                    avg_lap_time_millis = results_row["milliseconds"] / results_row["laps"]
            avg_lap_time_str = millis_to_str(avg_lap_time_millis)
            race_data["avg_lap_time_millis"].append(avg_lap_time_millis)
            race_data["avg_lap_time_str"].append(avg_lap_time_str)
            race_data["name"].append(name)
            race_data["constructor_name"].append(constructor_name)
            race_data["constructor_id"].append(constructor_id)
        race_data = pd.DataFrame.from_dict(race_data)
        if use_lap_data and race_laps.shape[0] == 0:
            race_data["rank"] = np.nan
        elif use_lap_data:
            race_data["rank"] = race_data["fastest_lap_time_millis"].rank(na_option="bottom").astype(int)
        race_data = race_data.sort_values(by="rank")
        race_data["rank"] = race_data["rank"].astype(str).replace("0", "").str.rjust(2)
        race_data["avg_lap_time_rank"] = race_data["avg_lap_time_millis"].rank()
        for idx, row in race_data.iterrows():
            fastest_lap_data.append([race_id] + row.values.tolist())
        if i % 100 == 0:
            print(f"Race {i} / {races.index.shape[0]}")
        i += 1
    fastest_lap_data = pd.DataFrame(data=fastest_lap_data, columns=["raceId", "driver_id", "name",
                                                                    "constructor_id", "constructor_name",
                                                                    "fastest_lap_time_millis", "fastest_lap_time_str",
                                                                    "avg_lap_time_millis", "avg_lap_time_str", "rank",
                                                                    "avg_lap_time_rank"])
    fastest_lap_data["rank"] = fastest_lap_data["rank"].apply(lambda x: str(x).rjust(2))  # TODO why do I do this??
    fastest_lap_data.to_csv("data/fastest_lap_data.csv", encoding="utf-8")

# ======================================================================================================================
# Save everything
# ======================================================================================================================
circuits.to_csv("temp_data/augmented/circuits_augmented.csv", encoding="utf-8")
constructor_results.to_csv("temp_data/augmented/constructor_results_augmented.csv", encoding="utf-8")
constructor_standings.to_csv("temp_data/augmented/constructor_standings_augmented.csv", encoding="utf-8")
constructors.to_csv("temp_data/augmented/constructors_augmented.csv", encoding="utf-8")
driver_standings.to_csv("temp_data/augmented/driver_standings_augmented.csv", encoding="utf-8")
drivers.to_csv("temp_data/augmented/drivers_augmented.csv", encoding="utf-8")
lap_times.to_csv("temp_data/augmented/lap_times_augmented.csv", encoding="utf-8")
pit_stops.to_csv("temp_data/augmented/pit_stops_augmented.csv", encoding="utf-8")
qualifying.to_csv("temp_data/augmented/qualifying_augmented.csv", encoding="utf-8")
races.to_csv("temp_data/augmented/races_augmented.csv", encoding="utf-8")
results.to_csv("temp_data/augmented/results_augmented.csv", encoding="utf-8")
seasons.to_csv("temp_data/augmented/seasons_augmented.csv", encoding="utf-8")
status.to_csv("temp_data/augmented/status_augmented.csv", encoding="utf-8")

# ======================================================================================================================
# Back-up the current data
# ======================================================================================================================
try:
    shutil.copyfile("data/circuits_augmented.csv", "backup_data/circuits_augmented.csv")
    shutil.copyfile("data/constructor_results_augmented.csv", "backup_data/constructor_results_augmented.csv")
    shutil.copyfile("data/constructor_standings_augmented.csv", "backup_data/constructor_standings_augmented.csv")
    shutil.copyfile("data/constructors_augmented.csv", "backup_data/constructors_augmented.csv")
    shutil.copyfile("data/driver_standings_augmented.csv", "backup_data/driver_standings_augmented.csv")
    shutil.copyfile("data/drivers_augmented.csv", "backup_data/drivers_augmented.csv")
    shutil.copyfile("data/lap_times_augmented.csv", "backup_data/lap_times_augmented.csv")
    shutil.copyfile("data/pit_stops_augmented.csv", "backup_data/pit_stops_augmented.csv")
    shutil.copyfile("data/qualifying_augmented.csv", "backup_data/qualifying_augmented.csv")
    shutil.copyfile("data/races_augmented.csv", "backup_data/races_augmented.csv")
    shutil.copyfile("data/results_augmented.csv", "backup_data/results_augmented.csv")
    shutil.copyfile("data/seasons_augmented.csv", "backup_data/seasons_augmented.csv")
    shutil.copyfile("data/status_augmented.csv", "backup_data/status_augmented.csv")
except:
    print("No data to back up!")

# ======================================================================================================================
# Save data to it's final destination
# ======================================================================================================================
circuits.to_csv("data/circuits_augmented.csv", encoding="utf-8")
constructor_results.to_csv("data/constructor_results_augmented.csv", encoding="utf-8")
constructor_standings.to_csv("data/constructor_standings_augmented.csv", encoding="utf-8")
constructors.to_csv("data/constructors_augmented.csv", encoding="utf-8")
driver_standings.to_csv("data/driver_standings_augmented.csv", encoding="utf-8")
drivers.to_csv("data/drivers_augmented.csv", encoding="utf-8")
lap_times.to_csv("data/lap_times_augmented.csv", encoding="utf-8")
pit_stops.to_csv("data/pit_stops_augmented.csv", encoding="utf-8")
qualifying.to_csv("data/qualifying_augmented.csv", encoding="utf-8")
races.to_csv("data/races_augmented.csv", encoding="utf-8")
results.to_csv("data/results_augmented.csv", encoding="utf-8")
seasons.to_csv("data/seasons_augmented.csv", encoding="utf-8")
status.to_csv("data/status_augmented.csv", encoding="utf-8")
