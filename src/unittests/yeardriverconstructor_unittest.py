import logging
import traceback
import time
import pandas as pd
import numpy as np
from data_loading.data_loader import load_results, load_races, load_drivers, load_constructors, load_fastest_lap_data, \
    load_driver_standings, load_constructor_standings
from mode.yeardriverconstructor import get_layout, is_valid_input, generate_wdc_plot, generate_wcc_plot, \
    generate_positions_plot, generate_spvfp_scatter, generate_mltr_fp_scatter, generate_win_plot, \
    generate_finishing_position_bar_plot, generate_teammatefp_fp_scatter, generate_teammate_diff_comparison_scatter, \
    generate_teammate_comparison_line_plot, generate_results_table, generate_stats_layout
from utils import time_decorator

# Biggest JOKE of a "unit test" but it works

error_ids = []
times = {
    "id": [],
    "time": []
}

logger = logging.getLogger()
logger.disabled = True

constructor_id = 6
driver_id = 8
year_id = 2015  # 2015 Kimi at Ferrari

results = load_results()
races = load_races()
driver_standings = load_driver_standings()
constructor_standings = load_constructor_standings()
fastest_lap_data = load_fastest_lap_data()
constructors = load_constructors()
drivers = load_drivers()

# Generate slices
year_races = races[races["year"] == year_id]
year_rids = year_races.index.values
year_results = results[results["raceId"].isin(year_rids)]
yd_results = year_results[year_results["driverId"] == driver_id]
yd_races = year_races.loc[yd_results["raceId"].values]
year_driver_standings = driver_standings[driver_standings["raceId"].isin(year_rids)]
yd_driver_standings = year_driver_standings[year_driver_standings["driverId"] == driver_id]
year_constructor_standings = constructor_standings[constructor_standings["raceId"].isin(year_rids)]
year_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"].isin(year_rids)]
yd_fastest_lap_data = year_fastest_lap_data[year_fastest_lap_data["driver_id"] == driver_id]
constructor_results = year_results[year_results["constructorId"] == constructor_id]

# WDC plot
generate_wdc_plot = time_decorator(generate_wdc_plot)
wdc_plot = generate_wdc_plot(year_races, year_driver_standings, year_results, driver_id, consider_window=2)

# WCC plot
generate_wcc_plot = time_decorator(generate_wcc_plot)
wcc_plot = generate_wcc_plot(year_races, year_constructor_standings, year_results, constructor_id)

# Positions plot
generate_positions_plot = time_decorator(generate_positions_plot)
positions_plot, positions_source = generate_positions_plot(yd_results, yd_driver_standings, yd_fastest_lap_data,
                                                           driver_id, year_id)

# Start pos vs finish pos scatter
generate_spvfp_scatter = time_decorator(generate_spvfp_scatter)
spvfp_scatter = generate_spvfp_scatter(yd_results, yd_races, yd_driver_standings)

# Mean lap time rank vs finish pos scatter
generate_mltr_fp_scatter = time_decorator(generate_mltr_fp_scatter)
mltr_fp_scatter = generate_mltr_fp_scatter(yd_results, yd_races, yd_driver_standings)

# Win plot
generate_win_plot = time_decorator(generate_win_plot)
win_plot = generate_win_plot(positions_source, driver_id)

# Finishing position bar plot
generate_finishing_position_bar_plot = time_decorator(generate_finishing_position_bar_plot)
position_dist = generate_finishing_position_bar_plot(yd_results)

# Teammate finish pos vs finish pos scatter
generate_teammatefp_fp_scatter = time_decorator(generate_teammatefp_fp_scatter)
teammatefp_fp_scatter = generate_teammatefp_fp_scatter(positions_source, constructor_results, driver_id)

# Teammate diff plot
generate_teammate_diff_comparison_scatter = time_decorator(generate_teammate_diff_comparison_scatter)
teammate_diff_plot, explanation_div, teammate_diff_source = generate_teammate_diff_comparison_scatter(
    positions_source, constructor_results, driver_id)

# Teammate comparison line plot
generate_teammate_comparison_line_plot = time_decorator(generate_teammate_comparison_line_plot)
teammate_comparison_line_plot, comparison_source = generate_teammate_comparison_line_plot(positions_source,
                                                                                          constructor_results,
                                                                                          driver_id)

# Results table
generate_results_table = time_decorator(generate_results_table)
results_table = generate_results_table(yd_results, yd_fastest_lap_data, year_results, year_fastest_lap_data,
                                       driver_id)

# Stats layout
generate_stats_layout = time_decorator(generate_stats_layout)
stats_layout = generate_stats_layout(positions_source, comparison_source, constructor_results, year_id, driver_id)

i = 0
invalid = 0
id_list = []
for constructor_id in constructors.index.unique():
    for driver_id in drivers.index.unique():
        for year_id in races["year"].unique():
            id_list.append((year_id, driver_id, constructor_id))
n_original = len(id_list)
pct_reduce = 0.02  # Out of 1
id_list = np.array(id_list)
id_list_idxs = np.random.choice(len(id_list), int(pct_reduce * n_original), replace=False)
id_list = id_list[id_list_idxs].tolist()
n = len(id_list)
print(f"NOTE: the length of id_list was reduced by {100 - pct_reduce * 100}% from {n_original} to {n} using random "
      f"subsampling in order to speed up the test")
for year_id, driver_id, constructor_id in id_list:
    try:
        i += 1
        if not is_valid_input(year_id, driver_id, constructor_id):
            invalid += 1
            if invalid % 500 == 0:
                print(str(round(100 * invalid / i, 2)) + "% of combos invalid currently")
            continue
        print(f"Testing year ID {year_id}, driver ID {driver_id}, constructor ID {constructor_id}, {i} / {n}")
        start = time.time()
        get_layout(year_id=year_id, driver_id=driver_id, constructor_id=constructor_id, download_image=False)
        end = time.time()
        elapsed = 1000 * (end - start)
        print(f"Completed in  {elapsed} milliseconds")
        times["id"].append((year_id, driver_id, constructor_id))
        times["time"].append(elapsed)
    except Exception as e:
        print(f"Encountered exception: {e}")
        track = traceback.format_exc()
        print("The traceback is:")
        print(track)
        error_ids.append((year_id, driver_id, constructor_id))
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
