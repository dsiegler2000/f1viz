import argparse
import pandas as pd
import os
import numpy as np
import re
from utils import str_to_millis

"""
Does basic data pre-processing.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--suppress_output", nargs="?", const=True, default=True, help="Suppresses output.")
args = parser.parse_args()


def print_check(df):
    if args.suppress_output:
        return
    print("NaN check:")
    print(df.isna().sum())
    print(r"\N check:")
    print(df[df == r"\N"].notna().sum())


def drop_after_2019(df, start_2020=1031):
    """
    Removes 2020 data by filtering by raceId.
    All raceId >= 1031 are 2020 races
    """
    df = df[df["raceId"] < start_2020]
    return df


def read_csv(filename):
    """
    Quick alias helper
    """
    return pd.read_csv(os.path.join("temp_data", "raw", filename))


def to_csv(df, filename):
    """
    Quick alias helper
    """
    df.to_csv(os.path.join("temp_data", "cleaned", filename), encoding="utf-8")


if __name__ == "__main__":
    # circuits.csv
    circuits = read_csv("circuits.csv")
    circuits["alt"] = circuits["alt"].replace(r"\N", np.nan).astype(np.float)
    to_csv(circuits, "circuits_cleaned.csv")
    print_check(circuits)

    # constructor_results.csv
    constructor_results = read_csv("constructor_results.csv")
    constructor_results = constructor_results.drop(columns=["status"])  # status column gives us nothing
    constructor_results = drop_after_2019(constructor_results)
    to_csv(constructor_results, "constructor_results_cleaned.csv")
    print_check(constructor_results)

    # constructor_standings.csv
    constructor_standings = read_csv("constructor_standings.csv")
    constructor_standings = drop_after_2019(constructor_standings)
    to_csv(constructor_standings, "constructor_standings_cleaned.csv")
    print_check(constructor_standings)

    # constructors.csv
    constructors = read_csv("constructors.csv")
    constructors = constructors.dropna(axis=1, how="all")  # drop that weird all NaN column
    to_csv(constructors, "constructors_cleaned.csv")
    print_check(constructors)

    # driver_standings.csv
    driver_standings = read_csv("driver_standings.csv")
    driver_standings = drop_after_2019(driver_standings)
    to_csv(driver_standings, "driver_standings_cleaned.csv")
    print_check(driver_standings)

    # drivers.csv
    drivers = read_csv("drivers.csv")
    drivers["number"] = drivers["number"].replace(r"\N", -1).astype(int)
    drivers["code"] = drivers["number"].fillna("XXX")
    drivers["dob"] = drivers["dob"].fillna(pd.Timestamp("19600101"))
    drivers["dob"] = pd.to_datetime(drivers["dob"])
    drivers["url"] = drivers["url"].fillna("https://en.wikipedia.org/wiki/Formula_One")
    to_csv(drivers, "drivers_cleaned.csv")
    print_check(drivers)

    # lap_times.csv
    lap_times = read_csv("lap_times.csv")
    lap_times = lap_times.drop(columns=["time"])
    to_csv(lap_times, "lap_times_cleaned.csv")
    print_check(lap_times)

    # pit_stops.csv
    pit_stops = read_csv("pit_stops.csv")
    pit_stops = drop_after_2019(pit_stops)
    to_csv(pit_stops, "pit_stops_cleaned.csv")
    print_check(pit_stops)

    # qualifying.csv
    qualifying = read_csv("qualifying.csv")
    qualifying["q1"] = qualifying["q1"].apply(str_to_millis).astype(float)
    qualifying["q2"] = qualifying["q2"].apply(str_to_millis).astype(float)
    qualifying["q3"] = qualifying["q3"].apply(str_to_millis).astype(float)
    qualifying = drop_after_2019(qualifying)
    to_csv(qualifying, "qualifying_cleaned.csv")
    print_check(qualifying)
    if not args.suppress_output:
        print(qualifying[qualifying["q1"].isna() & qualifying["q2"].isna() & qualifying["q3"].isna()])

    # races.csv
    races = read_csv("races.csv")
    races["time"] = races["time"].replace(r"\N", "00:00:00")
    races["date"] = races["date"].str.replace(re.escape(r"\N"), "")
    print_check(races)
    races["datetime"] = pd.to_datetime(races["date"].str.cat(races["time"], sep=" "))
    races["datetime"] = races["datetime"].fillna(pd.Timestamp("19900101"))
    races = races.drop(columns=["date", "time"])
    if races.index.name != "raceId":
        races = races.set_index("raceId")
    print_check(races)
    to_csv(races, "races_cleaned.csv")

    # results.csv
    results = read_csv("results.csv")
    results = results.drop(columns=["number"])
    results["position"] = results["position"].replace(r"\N", "")
    results["milliseconds"] = results["milliseconds"].replace(r"\N", None)
    results["fastestLap"] = results["fastestLap"].replace(r"\N", None)
    results["rank"] = results["rank"].replace(r"\N", None)
    results["fastestLapSpeed"] = results["fastestLapSpeed"].replace(r"\N", None)
    results["grid"] = results["grid"].replace(0, -1)
    results["fastestLapTime"] = results["fastestLapTime"].apply(str_to_millis)
    results["fastestLapSpeed"] = results["fastestLapSpeed"].astype(float, errors="ignore")
    results = results.drop(columns=["time"])
    results = drop_after_2019(results)
    to_csv(results, "results_cleaned.csv")
    print_check(results)

    # seasons.csv
    seasons = read_csv("seasons.csv")
    seasons = seasons[seasons["year"] < 2020]
    to_csv(seasons, "seasons_cleaned.csv")
    print_check(seasons)

    # status.csv
    status = read_csv("status.csv")
    to_csv(status, "status_cleaned.csv")
    print_check(status)
