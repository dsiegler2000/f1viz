import logging
import traceback
import time
import pandas as pd
from data_loading.data_loader import load_results, load_races, load_lap_times, load_circuits, \
    load_fastest_lap_data, load_driver_standings, load_constructor_standings, load_constructors
from mode.circuitconstructor import get_layout, generate_results_table, generate_lap_time_distribution_plot, \
    generate_win_plot, generate_positions_plot, is_valid_input, generate_finishing_position_bar_plot, \
    generate_spvfp_scatter, generate_mltr_fp_scatter, generate_stats_layout
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
constructors = load_constructors()
circuits = load_circuits()
races = load_races()
fastest_lap_data = load_fastest_lap_data()
lap_times = load_lap_times()
driver_standings = load_driver_standings()
constructor_standings = load_constructor_standings()

circuit_id = 6
constructor_id = 6  # Ferrari at Monaco

circuit_rids = races[races["circuitId"] == circuit_id].index.values
constructor_results = results[results["constructorId"] == constructor_id]
cc_results = constructor_results[constructor_results["raceId"].isin(circuit_rids)]
cc_rids = cc_results["raceId"]
cc_races = races[races.index.isin(cc_rids)]

# Generate some more slices
circuit_results = results[results["raceId"].isin(circuit_rids)]
circuit_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"].isin(circuit_rids)]
cc_years = cc_races["year"].unique()
cc_years.sort()
cc_fastest_lap_data_idxs = []
cc_lap_time_idxs = []
cc_driver_standings_idxs = []
for idx, results_row in cc_results.iterrows():
    rid = results_row["raceId"]
    did = results_row["driverId"]
    fl_slice = circuit_fastest_lap_data[circuit_fastest_lap_data["driver_id"] == did]
    cc_fastest_lap_data_idxs.extend(fl_slice.index.values.tolist())
    lt_slice = lap_times[(lap_times["raceId"] == rid) & (lap_times["driverId"] == did)]
    cc_lap_time_idxs.extend(lt_slice.index.values.tolist())
    driver_standings_slice = driver_standings[(driver_standings["raceId"] == rid) &
                                              (driver_standings["driverId"] == did)]
    cc_driver_standings_idxs.extend(driver_standings_slice.index.values.tolist())
cc_fastest_lap_data = circuit_fastest_lap_data.loc[cc_fastest_lap_data_idxs]
cc_lap_times = lap_times.loc[cc_lap_time_idxs]
cc_driver_standings = driver_standings.loc[cc_driver_standings_idxs]
cc_constructor_standings = constructor_standings[(constructor_standings["constructorId"] == constructor_id) &
                                                 (constructor_standings["raceId"].isin(cc_rids))]

# Positions plot
generate_positions_plot = time_decorator(generate_positions_plot)
positions_plot, positions_source = generate_positions_plot(cc_years, cc_fastest_lap_data, constructor_results,
                                                           cc_constructor_standings, cc_races, constructor_id)

# Win plot
generate_win_plot = time_decorator(generate_win_plot)
win_plot = generate_win_plot(positions_source, constructor_id)

# Lap time distribution plot
generate_lap_time_distribution_plot = time_decorator(generate_lap_time_distribution_plot)
lap_time_distribution_plot = generate_lap_time_distribution_plot(cc_lap_times, cc_rids, circuit_id, constructor_id)

# Finish position bar plot
generate_finishing_position_bar_plot = time_decorator(generate_finishing_position_bar_plot)
finish_position_bar_plot = generate_finishing_position_bar_plot(cc_results)

# Start pos vs finish pos scatter plot
generate_spvfp_scatter = time_decorator(generate_spvfp_scatter)
spvfp_scatter = generate_spvfp_scatter(cc_results, cc_races, cc_driver_standings)

# Mean lap time rank vs finish pos scatter plot
generate_mltr_fp_scatter = time_decorator(generate_mltr_fp_scatter)
mltr_fp_scatter = generate_mltr_fp_scatter(cc_results, cc_races, cc_driver_standings)

# Stats div
generate_stats_layout = time_decorator(generate_stats_layout)
stats_div = generate_stats_layout(cc_years, cc_races, cc_results, cc_fastest_lap_data, circuit_id, constructor_id)

# Results table
generate_results_table = time_decorator(generate_results_table)
results_table = generate_results_table(cc_results, cc_fastest_lap_data, circuit_results, circuit_fastest_lap_data)

i = 0
invalid = 0
id_list = []
for constructor_id in constructors.index.unique():
    for circuit_id in circuits.index.unique():
        id_list.append((circuit_id, constructor_id))
n = len(id_list)
for circuit_id, constructor_id in id_list:
    try:
        i += 1
        if not is_valid_input(circuit_id, constructor_id):
            invalid += 1
            print(str(round(100 * invalid / i, 1)) + "% of combos invalid currently")
            continue
        print(f"Testing circuit ID {circuit_id}, constructor ID {constructor_id}, {i} / {n}")
        start = time.time()
        get_layout(circuit_id=circuit_id, constructor_id=constructor_id, download_image=False)
        end = time.time()
        elapsed = 1000 * (end - start)
        print(f"Completed in  {elapsed} milliseconds")
        times["id"].append((circuit_id, constructor_id))
        times["time"].append(elapsed)
    except Exception as e:
        print(f"Encountered exception: {e}")
        track = traceback.format_exc()
        print("The traceback is:")
        print(track)
        error_ids.append((circuit_id, constructor_id))
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
