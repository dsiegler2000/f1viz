import logging
import traceback
import time
import pandas as pd
from data_loading.data_loader import load_races, load_constructors, load_status, load_fastest_lap_data, \
    load_driver_standings, load_constructor_standings, load_results
from mode.yearconstructor import get_layout, is_valid_input, generate_wcc_plot, generate_positions_plot, \
    generate_spvfp_scatter, generate_win_plot, generate_finishing_position_bar_plot, generate_driver_performance_table, \
    generate_results_table, generate_teammate_comparison_line_plot, generate_mltr_fp_scatter, generate_stats_layout
from utils import time_decorator, get_constructor_name

# Biggest JOKE of a "unit test" but it works

error_ids = []
times = {
    "id": [],
    "time": []
}

logger = logging.getLogger()
logger.disabled = True

results = load_results()
constructor_standings = load_constructor_standings()
driver_standings = load_driver_standings()
races = load_races()
fastest_lap_data = load_fastest_lap_data()
status = load_status()
constructors = load_constructors()

year_id = 2014
constructor_id = 6  # Ferrari in 2014

year_races = races[races["year"] == year_id]
year_results = results[results["raceId"].isin(year_races.index)]
yc_results = year_results[year_results["constructorId"] == constructor_id]
year_constructor_standings = constructor_standings[constructor_standings["raceId"].isin(year_races.index)]
year_constructor_standings = year_constructor_standings.sort_values(by="raceId")
yc_constructor_standings = year_constructor_standings[year_constructor_standings["constructorId"] == constructor_id]
year_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"].isin(year_races.index)]
yc_fastest_lap_data_idxs = []
yc_driver_standings_idxs = []
for idx, row in yc_results.iterrows():
    rid = row["raceId"]
    did = row["driverId"]
    fl_slice = year_fastest_lap_data[(year_fastest_lap_data["raceId"] == rid) &
                                     (year_fastest_lap_data["driver_id"] == did)]
    yc_fastest_lap_data_idxs.extend(fl_slice.index.values.tolist())
    driver_standings_slice = driver_standings[(driver_standings["raceId"] == rid) &
                                              (driver_standings["driverId"] == did)]
    yc_driver_standings_idxs.extend(driver_standings_slice.index.values.tolist())
yc_fastest_lap_data = fastest_lap_data.loc[yc_fastest_lap_data_idxs]
yc_driver_standings = driver_standings.loc[yc_driver_standings_idxs]
yc_races = year_races[year_races.index.isin(yc_results["raceId"])]

# WCC plot
generate_wcc_plot = time_decorator(generate_wcc_plot)
wcc_plot = generate_wcc_plot(year_races, year_constructor_standings, year_results, constructor_id)

# Positions plot
generate_positions_plot = time_decorator(generate_positions_plot)
positions_plot, positions_source = generate_positions_plot(yc_constructor_standings, yc_results,
                                                           yc_fastest_lap_data, year_id, constructor_id)

# Start pos v finish pos scatter
generate_spvfp_scatter = time_decorator(generate_spvfp_scatter)
spvfp_scatter = generate_spvfp_scatter(yc_results, yc_races, yc_driver_standings)

# Mean lap time rank vs finish pos scatter
generate_mltr_fp_scatter = time_decorator(generate_mltr_fp_scatter)
mltr_fp_scatter = generate_mltr_fp_scatter(yc_results, yc_races, yc_driver_standings)

# Win plot
generate_win_plot = time_decorator(generate_win_plot)
win_plot = generate_win_plot(positions_source, constructor_id)

# Finish pos bar plot
generate_finishing_position_bar_plot = time_decorator(generate_finishing_position_bar_plot)
finishing_position_bar_plot = generate_finishing_position_bar_plot(yc_results)

# Driver performance table
generate_driver_performance_table = time_decorator(generate_driver_performance_table)
driver_performance_layout = generate_driver_performance_table(yc_races, yc_results)

# Results table
generate_results_table = time_decorator(generate_results_table)
results_table, results_source = generate_results_table(yc_results, yc_fastest_lap_data, year_results,
                                                       year_fastest_lap_data)

# Teammate comparison line plot
generate_teammate_comparison_line_plot = time_decorator(generate_teammate_comparison_line_plot)
teammate_comparison_line_plot, comparison_source = generate_teammate_comparison_line_plot(yc_results, year_races,
                                                                                          yc_driver_standings,
                                                                                          year_id)

# Stats layout
generate_stats_layout = time_decorator(generate_stats_layout)
stats_layout = generate_stats_layout(positions_source, yc_results, comparison_source, year_id, constructor_id)

i = 0
invalid = 0
id_list = []
for year_id in races["year"].unique():
    for constructor_id in constructors.index.unique():
        id_list.append((year_id, constructor_id))
n = len(id_list)
for year_id, constructor_id in id_list:
    try:
        i += 1
        if not is_valid_input(year_id, constructor_id):
            invalid += 1
            if invalid % 100 == 0:
                print(str(round(100 * invalid / i, 1)) + "% of combos invalid currently")
            continue
        constructor_name = get_constructor_name(constructor_id, include_flag=False)
        print(f"Testing year ID {year_id}, constructor ID {constructor_id}, {i} / {n}")
        start = time.time()
        get_layout(year_id=year_id, constructor_id=constructor_id)
        end = time.time()
        elapsed = 1000 * (end - start)
        print(f"Completed in  {elapsed} milliseconds")
        times["id"].append((year_id, constructor_id))
        times["time"].append(elapsed)
    except Exception as e:
        print(f"Encountered exception: {e}")
        trace = traceback.format_exc()
        print("The traceback is:")
        print(trace)
        error_ids.append((year_id, constructor_id))
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
