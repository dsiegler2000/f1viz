import logging
import traceback
from data_loading.data_loader import load_races, load_seasons, load_driver_standings, load_constructor_standings, \
    load_results, load_lap_times, load_qualifying, load_fastest_lap_data
from mode.year import get_layout, generate_wdc_plot, generate_wcc_plot, \
    generate_position_mlt_scatter, generate_position_mfms_scatter, generate_teams_and_drivers, generate_races_info, \
    generate_wdc_results, generate_dnf_table
import time
import pandas as pd

# Biggest JOKE of a "unit test" but it works
from utils import time_decorator

races = load_races()
error_ids = []
times = {
    "yid": [],
    "time": []
}

logger = logging.getLogger()
logger.disabled = True

seasons = load_seasons()
driver_standings = load_driver_standings()
constructor_standings = load_constructor_standings()
results = load_results()
lap_times = load_lap_times()
qualifying = load_qualifying()
fastest_lap_data = load_fastest_lap_data()
year_id = 2003

# Generate some useful slices
year_races = races[races["year"] == year_id]
year_driver_standings = driver_standings[driver_standings["raceId"].isin(year_races.index)].sort_values(by="raceId")
year_constructor_standings = constructor_standings[constructor_standings["raceId"].isin(year_races.index)].sort_values(by="raceId")
year_results = results[results["raceId"].isin(year_races.index)]
year_laps = lap_times[lap_times["raceId"].isin(year_races.index)]
year_qualifying = qualifying[qualifying["raceId"].isin(year_races.index)]
year_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"].isin(year_races.index)]

logging.info(f"Generating layout for mode YEAR in year, year_id={year_id}")

generate_wdc_plot = time_decorator(generate_wdc_plot)
generate_wdc_plot(year_driver_standings, year_results)

generate_constructors_championship_plot = time_decorator(generate_wcc_plot)
generate_constructors_championship_plot(year_constructor_standings, year_results)

generate_position_mlt_scatter = time_decorator(generate_position_mlt_scatter)
generate_position_mlt_scatter(year_laps, year_results, year_driver_standings, year_constructor_standings)

generate_position_mfms_scatter = time_decorator(generate_position_mfms_scatter)
generate_position_mfms_scatter(year_results, year_driver_standings)

generate_teams_and_drivers = time_decorator(generate_teams_and_drivers)
generate_teams_and_drivers(year_results, year_races)

generate_races_info = time_decorator(generate_races_info)
generate_races_info(year_races, year_qualifying, year_results, year_fastest_lap_data)

generate_wdc_results = time_decorator(generate_wdc_results)
generate_wdc_results(year_results, year_driver_standings, year_races)

generate_dnf_table = time_decorator(generate_dnf_table)
generate_dnf_table(year_results)

n = races["year"].unique().shape[0]
i = 1

# 1950 - 1957 didn't have a WDC
for year_id in races["year"].unique():
    try:
        print(f"Testing year ID {year_id}, {i} / {n}")
        i += 1
        start = time.time()
        get_layout(year_id=year_id)
        end = time.time()
        elapsed = 1000 * (end - start)
        print(f"Completed in  {elapsed} milliseconds")
        times["yid"].append(year_id)
        times["time"].append(elapsed)
    except Exception as e:
        print(f"Encountered exception: {e}")
        track = traceback.format_exc()
        print("The traceback is:")
        print(track)
        error_ids.append(year_id)
    print("=======================================")

print("The following year IDs had errors: ")
print(error_ids)
print("Times:")
print(times)
times = pd.DataFrame.from_dict(times)
print(times["time"].describe())
print("Outliers:")
print(times[times["time"] > times["time"].std() + times["time"].mean()])
times.to_csv("src/unittests/times.csv")
print("Saved to src/unittests/times.csv")
