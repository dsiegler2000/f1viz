import logging
import traceback
import time
import pandas as pd
import numpy as np
from data_loading.data_loader import load_results, load_races, load_lap_times, load_drivers, load_circuits, \
    load_pit_stops, load_qualifying, load_fastest_lap_data, load_driver_standings
from mode.yearcircuitdriver import get_layout, is_valid_input, generate_gap_plot, generate_position_plot, \
    generate_lap_time_plot, detect_mark_safety_car, mark_fastest_lap, detect_mark_overtakes, mark_pit_stops, \
    generate_quali_table, generate_stats_layout
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
drivers = load_drivers()
circuits = load_circuits()
results = load_results()
lap_times = load_lap_times()
pit_stop_data = load_pit_stops()
quali = load_qualifying()
fastest_lap_data = load_fastest_lap_data()
driver_standings = load_driver_standings()

driver_id = 8
circuit_id = 6
year_id = 2015  # 2015 Kimi at Monaco

year_races = races[races["year"] == year_id]
race = races[(races["circuitId"] == circuit_id) & (races["year"] == year_id)]
rid = race.index.values[0]
race_results = results[results["raceId"] == rid]
ycd_results = race_results[race_results["driverId"] == driver_id]
race_laps = lap_times[lap_times["raceId"] == rid]
ycd_laps = race_laps[race_laps["driverId"] == driver_id]
ycd_pit_stop_data = pit_stop_data[(pit_stop_data["raceId"] == rid) & (pit_stop_data["driverId"] == driver_id)]
race_quali = quali[quali["raceId"] == rid]
ycd_fastest_lap_data = fastest_lap_data[(fastest_lap_data["raceId"] == rid) &
                                        (fastest_lap_data["driver_id"] == driver_id)]
year_driver_standings = driver_standings[driver_standings["raceId"].isin(year_races.index.values)]

# Gap plot
generate_gap_plot = time_decorator(generate_gap_plot)
gap_plot, cached_driver_map = generate_gap_plot(race_laps, race_results, driver_id)

# Position plot
generate_position_plot = time_decorator(generate_position_plot)
position_plot = generate_position_plot(race_laps, race_results, cached_driver_map, driver_id)

# Lap time plot
generate_lap_time_plot = time_decorator(generate_lap_time_plot)
lap_time_plot = generate_lap_time_plot(race_laps, race_results, cached_driver_map, driver_id)

plots = [gap_plot, position_plot, lap_time_plot]

# Mark safety car
detect_mark_safety_car = time_decorator(detect_mark_safety_car)
disclaimer_sc = detect_mark_safety_car(race_laps, race, race_results, plots)

# Mark fastest lap
mark_fastest_lap = time_decorator(mark_fastest_lap)
mark_fastest_lap(ycd_results, plots)

plots = [gap_plot, lap_time_plot]

# Mark overtakes
detect_mark_overtakes = time_decorator(detect_mark_overtakes)
disclaimer_overtakes = detect_mark_overtakes(ycd_laps, race_laps, plots)

# Mark pit stops
mark_pit_stops = time_decorator(mark_pit_stops)
mark_pit_stops(ycd_pit_stop_data, [gap_plot, lap_time_plot], driver_id)

# Quali table
generate_quali_table = time_decorator(generate_quali_table)
quali_table, quali_source = generate_quali_table(race_quali, race_results, driver_id)

# Stats layout
generate_stats_layout = time_decorator(generate_stats_layout)
stats_layout = generate_stats_layout(ycd_results, ycd_pit_stop_data, ycd_fastest_lap_data, year_driver_standings,
                                     race_results, quali_source, rid, circuit_id, driver_id)

i = 0
invalid = 0
id_list = []
for driver_id in drivers.index.unique():
    for circuit_id in circuits.index.unique():
        for year_id in races["year"].unique():
            id_list.append((year_id, circuit_id, driver_id))
n_original = len(id_list)
pct_reduce = 0.07  # Out of 1
id_list = np.array(id_list)
id_list_idxs = np.random.choice(len(id_list), int(pct_reduce * n_original), replace=False)
id_list = id_list[id_list_idxs].tolist()
n = len(id_list)
print(f"NOTE: the length of id_list was reduced by {100 - pct_reduce * 100}% from {n_original} to {n} using random "
      f"subsampling in order to speed up the test")
for year_id, circuit_id, driver_id in id_list:
    try:
        i += 1
        if not is_valid_input(year_id, circuit_id, driver_id):
            invalid += 1
            if invalid % 100 == 0:
                print(str(round(100 * invalid / i, 1)) + "% of combos invalid currently")
            continue
        print(f"Testing year ID {year_id}, circuit ID {circuit_id}, driver ID {driver_id}, {i} / {n}")
        start = time.time()
        get_layout(year_id=year_id, circuit_id=circuit_id, driver_id=driver_id, download_image=False)
        end = time.time()
        elapsed = 1000 * (end - start)
        print(f"Completed in  {elapsed} milliseconds")
        times["id"].append((year_id, circuit_id, driver_id))
        times["time"].append(elapsed)
    except Exception as e:
        print(f"Encountered exception: {e}")
        track = traceback.format_exc()
        print("The traceback is:")
        print(track)
        error_ids.append((year_id, circuit_id, driver_id))
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
