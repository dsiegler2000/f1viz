import logging
import traceback
import time
import pandas as pd
from data_loading.data_loader import load_constructors, load_drivers, load_results, load_driver_standings, load_races, \
    load_fastest_lap_data, load_constructor_standings
from mode.driverconstructor import is_valid_input, get_layout, generate_positions_plot, mark_teammate_changes, \
    generate_win_plot, generate_teammatefp_fp_scatter, generate_finishing_position_bar_plot, \
    generate_wdc_position_bar_plot, generate_wcc_position_bar_plot, generate_spvfp_scatter, generate_mltr_fp_scatter, \
    generate_circuit_performance_table, generate_teammate_diff_comparison_scatter, generate_stats_layout, \
    generate_teammate_comparison_line_plot
from utils import time_decorator, get_driver_name, get_constructor_name

# Biggest JOKE of a "unit test" but it works

error_ids = []
times = {
    "id": [],
    "time": []
}

logger = logging.getLogger()
logger.disabled = True

drivers = load_drivers()
constructors = load_constructors()
results = load_results()
driver_standings = load_driver_standings()
races = load_races()
fastest_lap_data = load_fastest_lap_data()
constructor_standings = load_constructor_standings()

driver_id = 8
constructor_id = 6  # Kimi at Ferrari

constructor_results = results[results["constructorId"] == constructor_id]
dc_results = constructor_results[constructor_results["driverId"] == driver_id]
dc_rids = dc_results["raceId"].unique()
driver_driver_standings = driver_standings[driver_standings["driverId"] == driver_id]
dc_driver_standings = driver_driver_standings[driver_driver_standings["raceId"].isin(dc_rids)]
dc_races = races[races.index.isin(dc_rids)].sort_values(by=["year", "raceId"])
dc_years = dc_races["year"].unique()
dc_fastest_lap_data = fastest_lap_data[(fastest_lap_data["driver_id"] == driver_id) &
                                       (fastest_lap_data["raceId"].isin(dc_rids))]
constructor_constructor_standings = constructor_standings[constructor_standings["constructorId"] == constructor_id]

# Positions plot
generate_positions_plot = time_decorator(generate_positions_plot)
positions_plot, positions_source = generate_positions_plot(dc_years, dc_driver_standings, dc_results,
                                                           dc_fastest_lap_data, driver_id, constructor_id)

# Mark teammate change
mark_teammate_changes = time_decorator(mark_teammate_changes)
mark_teammate_changes(positions_source, constructor_results, driver_id, positions_plot)

# Win plot
generate_win_plot = time_decorator(generate_win_plot)
win_plot = generate_win_plot(positions_source)

# Teammate finish pos vs driver finish pos scatter
generate_teammatefp_fp_scatter = time_decorator(generate_teammatefp_fp_scatter)
teammatefp_fp_scatter = generate_teammatefp_fp_scatter(positions_source, constructor_results, driver_id)

# Teammate diff plot
generate_teammate_diff_comparison_scatter = time_decorator(generate_teammate_diff_comparison_scatter)
teammate_diff_plot, explanation_div, teammate_diff_source = generate_teammate_diff_comparison_scatter(
    positions_source, constructor_results, driver_id)

# Teammate finish pos vs driver finish pos line plot
generate_teammate_comparison_line_plot = time_decorator(generate_teammate_comparison_line_plot)
teammate_comparison_line = generate_teammate_comparison_line_plot(positions_source, constructor_results, driver_id)

# Finishing position bar plot
generate_finishing_position_bar_plot = time_decorator(generate_finishing_position_bar_plot)
finishing_position_bar_plot = generate_finishing_position_bar_plot(dc_results)

# WDC position bar plot
generate_wdc_position_bar_plot = time_decorator(generate_wdc_position_bar_plot)
wdc_position_bar_plot = generate_wdc_position_bar_plot(positions_source)

# WCC position bar plot
generate_wcc_position_bar_plot = time_decorator(generate_wcc_position_bar_plot)
wcc_position_bar_plot, wcc_position_source = generate_wcc_position_bar_plot(dc_years,
                                                                            constructor_constructor_standings)

# Start pos. vs finish pos. scatter
generate_spvfp_scatter = time_decorator(generate_spvfp_scatter)
spvfp_scatter = generate_spvfp_scatter(dc_results, dc_races, driver_driver_standings)

# Mean lap time rank vs finish pos. scatter
generate_mltr_fp_scatter = time_decorator(generate_mltr_fp_scatter)
mltr_fp_scatter = generate_mltr_fp_scatter(dc_results, dc_races, driver_driver_standings, driver_id)

# Circuit performance table
generate_circuit_performance_table = time_decorator(generate_circuit_performance_table)
circuit_performance_table = generate_circuit_performance_table(dc_results, dc_races, driver_id, constructor_id)

# Stats
generate_stats_layout = time_decorator(generate_stats_layout)
stats_layout = generate_stats_layout(dc_years, dc_races, dc_results, positions_source, wcc_position_source,
                                     teammate_diff_source, driver_id, constructor_id)

i = 0
invalid = 0
id_list = []
for driver_id in drivers.index.unique():
    for constructor_id in constructors.index.unique():
        id_list.append((driver_id, constructor_id))
n = len(id_list)
for driver_id, constructor_id in id_list:
    try:
        i += 1
        if not is_valid_input(driver_id, constructor_id):
            invalid += 1
            if invalid % 500 == 0:
                print(str(round(100 * invalid / i, 1)) + "% of combos invalid currently")
            continue
        driver_name = get_driver_name(driver_id, include_flag=False, just_last=True)
        constructor_name = get_constructor_name(constructor_id, include_flag=False)
        print(f"Testing driver ID {driver_id}, {driver_name}, constructor ID {constructor_id}, {constructor_name}, "
              f"{i} / {n}")
        start = time.time()
        get_layout(driver_id=driver_id, constructor_id=constructor_id, download_image=False)
        end = time.time()
        elapsed = 1000 * (end - start)
        print(f"Completed in  {elapsed} milliseconds")
        times["id"].append((driver_id, constructor_id))
        times["time"].append(elapsed)
    except Exception as e:
        print(f"Encountered exception: {e}")
        trace = traceback.format_exc()
        print("The traceback is:")
        print(trace)
        error_ids.append((driver_id, constructor_id))
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
