import logging
import traceback
import time
import pandas as pd
import numpy as np
from data_loading.data_loader import load_results, load_races, load_lap_times, load_drivers, load_circuits, \
    load_pit_stops, load_constructors, load_qualifying, load_fastest_lap_data, load_driver_standings, \
    load_constructor_standings
from mode.yearcircuitconstructor import generate_gap_plot, generate_position_plot, generate_lap_time_plot, \
    detect_mark_safety_car, mark_pit_stops, get_layout, is_valid_input, generate_quali_table, generate_stats_layout
from utils import time_decorator

# Biggest JOKE of a "unit test" but it works

error_ids = []
times = {
    "id": [],
    "time": []
}

logger = logging.getLogger()
logger.disabled = True

races = load_races()
results = load_results()
lap_times = load_lap_times()
pit_stop_data = load_pit_stops()
quali = load_qualifying()
fastest_lap_data = load_fastest_lap_data()
driver_standings = load_driver_standings()
constructor_standings = load_constructor_standings()
constructors = load_constructors()
circuits = load_circuits()
drivers = load_drivers()

constructor_id = 6
circuit_id = 6
year_id = 2015  # 2015 Ferrari at Monaco

year_races = races[races["year"] == year_id]
race = year_races[year_races["circuitId"] == circuit_id]
rid = race.index.values[0]
race_results = results[results["raceId"] == rid]
ycc_results = race_results[race_results["constructorId"] == constructor_id]
driver_ids = ycc_results["driverId"].unique()
ycc_pit_stop_data = pit_stop_data[(pit_stop_data["raceId"] == rid) & (pit_stop_data["driverId"].isin(driver_ids))]
race_laps = lap_times[lap_times["raceId"] == rid]
race_quali = quali[quali["raceId"] == rid]
ycc_fastest_lap_data = fastest_lap_data[(fastest_lap_data["raceId"] == rid) &
                                        (fastest_lap_data["driver_id"].isin(driver_ids))]
year_driver_standings = driver_standings[driver_standings["raceId"].isin(year_races.index.values)]
year_constructor_standings = constructor_standings[constructor_standings["raceId"].isin(year_races.index.values)]

# Gap plot
generate_gap_plot = time_decorator(generate_gap_plot)
gap_plot, cached_driver_map = generate_gap_plot(race_laps, race_results, driver_ids, constructor_id)

# Position plot
generate_position_plot = time_decorator(generate_position_plot)
position_plot = generate_position_plot(race_laps, race_results, cached_driver_map, driver_ids, constructor_id)

# Lap time plot
generate_lap_time_plot = time_decorator(generate_lap_time_plot)
lap_time_plot = generate_lap_time_plot(race_laps, race_results, cached_driver_map, driver_ids, constructor_id)

plots = [gap_plot, position_plot, lap_time_plot]

# Mark safety car
detect_mark_safety_car = time_decorator(detect_mark_safety_car)
disclaimer_sc = detect_mark_safety_car(race_laps, race, race_results, plots)

# Mark pit stops
mark_pit_stops = time_decorator(mark_pit_stops)
mark_pit_stops(ycc_pit_stop_data, driver_ids, cached_driver_map, plots)

# Quali table
generate_quali_table = time_decorator(generate_quali_table)
quali_table, quali_source = generate_quali_table(race_quali, race_results, driver_ids)

# Stats layout
generate_stats_layout = time_decorator(generate_stats_layout)
stats_layout = generate_stats_layout(ycc_results, ycc_pit_stop_data, ycc_fastest_lap_data, year_driver_standings,
                                     year_constructor_standings, quali_source, rid, circuit_id, constructor_id,
                                     driver_ids)

i = 0
invalid = 0
id_list = []
for constructor_id in constructors.index.unique():
    for circuit_id in circuits.index.unique():
        for year_id in races["year"].unique():
            id_list.append((year_id, circuit_id, constructor_id))
n_original = len(id_list)
pct_reduce = 0.15  # Out of 1
id_list = np.array(id_list)
id_list_idxs = np.random.choice(len(id_list), int(pct_reduce * n_original), replace=False)
id_list = id_list[id_list_idxs].tolist()
n = len(id_list)
print(f"NOTE: the length of id_list was reduced by {100 - pct_reduce * 100}% from {n_original} to {n} using random "
      f"subsampling in order to speed up the test")
for year_id, circuit_id, constructor_id in id_list:
    try:
        i += 1
        if not is_valid_input(year_id, circuit_id, constructor_id):
            invalid += 1
            if invalid % 100 == 0:
                print(str(round(100 * invalid / i, 1)) + "% of combos invalid currently")
            continue
        print(f"Testing year ID {year_id}, circuit ID {circuit_id}, constructor ID {constructor_id}, {i} / {n}")
        start = time.time()
        get_layout(year_id=year_id, circuit_id=circuit_id, constructor_id=constructor_id, download_image=False)
        end = time.time()
        elapsed = 1000 * (end - start)
        print(f"Completed in  {elapsed} milliseconds")
        times["id"].append((year_id, circuit_id, constructor_id))
        times["time"].append(elapsed)
    except Exception as e:
        print(f"Encountered exception: {e}")
        track = traceback.format_exc()
        print("The traceback is:")
        print(track)
        error_ids.append((year_id, circuit_id, constructor_id))
    print("=======================================")

print("The following IDs had errors: ")
print(error_ids)
print("Times:")
print(times)
times = pd.DataFrame.from_dict(times)
print(times["time"].describe())
print("Outliers:")
print(times[times["time"] > times["time"].std() + times["time"].mean()])
times.to_csv("src/unittests/times.csv")
print("Saved to times.csv")
