import logging
import traceback
import time
import pandas as pd
import numpy as np
from data_loading.data_loader import load_results, load_races, load_lap_times, load_drivers, load_circuits, \
    load_fastest_lap_data, load_driver_standings, load_constructor_standings
from mode.circuitdriver import get_layout, generate_lap_time_plot, is_valid_input, generate_results_table, \
    generate_positions_plot, generate_spvfp_scatter, generate_mltr_fp_scatter, generate_stats_layout
from utils import time_decorator

# Biggest JOKE of a "unit test" but it works

error_ids = []
times = {
    "id": [],
    "time": []
}

logger = logging.getLogger()
logger.disabled = True

results = load_results()
races = load_races()
lap_times = load_lap_times()
drivers = load_drivers()
circuits = load_circuits()
fastest_lap_data = load_fastest_lap_data()
driver_standings = load_driver_standings()
constructor_standings = load_constructor_standings()

driver_id = 8
circuit_id = 6  # Kimi at Monaco
circuit_rids = races[races["circuitId"] == circuit_id].index.values
cd_results = results[(results["raceId"].isin(circuit_rids)) & (results["driverId"] == driver_id)]
cd_rids = cd_results["raceId"]
cd_races = races[races.index.isin(cd_rids)]
cd_lap_times = lap_times[(lap_times["raceId"].isin(cd_rids)) & (lap_times["driverId"] == driver_id)]
cd_years = cd_races["year"].unique()
cd_years.sort()
cd_fastest_lap_data = fastest_lap_data[(fastest_lap_data["raceId"].isin(cd_rids)) &
                                       (fastest_lap_data["driver_id"] == driver_id)]
driver_results = results[results["driverId"] == driver_id]
circuit_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"].isin(circuit_rids)]
driver_driver_standings = driver_standings[driver_standings["driverId"] == driver_id]
cd_constructor_standings_idxs = []
for idx, results_row in cd_results.iterrows():
    rid = results_row["raceId"]
    cid = results_row["constructorId"]
    constructor_standings_slice = constructor_standings[(constructor_standings["raceId"] == rid) &
                                                        (constructor_standings["constructorId"] == cid)]
    cd_constructor_standings_idxs.extend(constructor_standings_slice.index.values.tolist())
cd_constructor_standings = constructor_standings.loc[cd_constructor_standings_idxs]

# Positions plot
generate_positions_plot = time_decorator(generate_positions_plot)
positions_plot, positions_source = generate_positions_plot(cd_years, driver_driver_standings, driver_results,
                                                           cd_fastest_lap_data, cd_races, driver_id)

# Lap time distribution plot
generate_lap_time_distribution_plot = time_decorator(generate_lap_time_plot)
lap_time_dist = generate_lap_time_distribution_plot(cd_lap_times, cd_rids, driver_id, circuit_id)

# Starting position vs finish position scatter
generate_spvfp_scatter = time_decorator(generate_spvfp_scatter)
spvfp_scatter = generate_spvfp_scatter(cd_results, cd_races, driver_driver_standings)

# Mean lap time rank vs finish position scatter plot
generate_mltr_fp_scatter = time_decorator(generate_mltr_fp_scatter)
mltr_fp_scatter = generate_mltr_fp_scatter(cd_results, cd_races, driver_driver_standings, driver_id)

# Results table
generate_results_table = time_decorator(generate_results_table)
results_table = generate_results_table(cd_years, cd_races, cd_results, cd_fastest_lap_data)

# Stats div
generate_stats_div = time_decorator(generate_stats_layout)
stats_div = generate_stats_div(cd_years, cd_races, cd_results, cd_fastest_lap_data, driver_id, circuit_id)

i = 0
invalid = 0
id_list = []
for driver_id in drivers.index.unique():
    for circuit_id in circuits.index.unique():
        id_list.append((circuit_id, driver_id))
n_original = len(id_list)
pct_reduce = 0.07  # Out of 1
id_list = np.array(id_list)
id_list_idxs = np.random.choice(len(id_list), int(pct_reduce * n_original), replace=False)
id_list = id_list[id_list_idxs].tolist()
n = len(id_list)
print(f"NOTE: the length of id_list was reduced by {100 - pct_reduce * 100}% from {n_original} to {n} using random "
      f"subsampling in order to speed up the test")
for circuit_id, driver_id in id_list:
    try:
        i += 1
        if not is_valid_input(circuit_id, driver_id):
            invalid += 1
            print(str(round(100 * invalid / i, 1)) + "% of combos invalid currently")
            continue
        print(f"Testing circuit ID {circuit_id}, driver ID {driver_id}, {i} / {n}")
        start = time.time()
        get_layout(circuit_id=circuit_id, driver_id=driver_id, download_image=False)
        end = time.time()
        elapsed = 1000 * (end - start)
        print(f"Completed in  {elapsed} milliseconds")
        times["id"].append((circuit_id, driver_id))
        times["time"].append(elapsed)
    except Exception as e:
        print(f"Encountered exception: {e}")
        track = traceback.format_exc()
        print("The traceback is:")
        print(track)
        error_ids.append((circuit_id, driver_id))
    print("=======================================")

print("The following race IDs had errors: ")
print(error_ids)
print("Times:")
print(times)
times = pd.DataFrame.from_dict(times)
print(times["time"].describe())
print("Outliers:")
print(times[times["time"] > times["time"].std() + times["time"].mean()])
times.to_csv("src/unittests/times.csv")
print("Saved to times.csv")
