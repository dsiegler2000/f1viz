import logging
import traceback
import time
import pandas as pd
from mode.driver import get_layout, generate_team_performance_layout, generate_spvfp_scatter, generate_mltr_fp_scatter
from data_loading.data_loader import load_drivers, load_races, load_driver_standings, load_results, \
    load_fastest_lap_data
from mode.driver import generate_win_plot, generate_positions_plot, mark_team_changes, \
    generate_circuit_performance_table, generate_finishing_position_bar_plot, generate_wdc_position_bar_plot
from utils import time_decorator, get_driver_name

# Biggest JOKE of a "unit test" but it works
drivers = load_drivers()
error_ids = []
times = {
    "did": [],
    "time": []
}

logger = logging.getLogger()
logger.disabled = True

driver_standings = load_driver_standings()
results = load_results()
fastest_lap_data = load_fastest_lap_data()
races = load_races()
driver_id = 30  # Schumacher

driver_results = results[results["driverId"] == driver_id]
driver_driver_standings = driver_standings[driver_standings["driverId"] == driver_id]
driver_rids = driver_results["raceId"].unique()
driver_races = races[races.index.isin(driver_rids)].sort_values(by=["year", "raceId"])
driver_years = driver_races["year"].unique()
driver_fastest_lap_data = fastest_lap_data[fastest_lap_data["driver_id"] == driver_id]

logging.info(f"Generating layout for mode DRIVER in driver, driver_id={driver_id}")

# Position plot
generate_positions_plot = time_decorator(generate_positions_plot)
positions_plot, positions_source = generate_positions_plot(driver_years, driver_driver_standings, driver_results,
                                                           driver_fastest_lap_data, driver_id)

# Mark constructor changes
mark_team_changes = time_decorator(mark_team_changes)
mark_team_changes(driver_years, driver_results, [positions_plot], [positions_source])

# Circuit performance table
generate_circuit_performance_table = time_decorator(generate_circuit_performance_table)
circuit_performance_table = generate_circuit_performance_table(driver_results, driver_races, driver_id)

# Position bar plot
generate_finishing_position_bar_plot = time_decorator(generate_finishing_position_bar_plot)
position_dist = generate_finishing_position_bar_plot(driver_results)

# WDC Position bar plot
generate_wdc_position_bar_plot = time_decorator(generate_wdc_position_bar_plot)
wdc_position_dist = generate_wdc_position_bar_plot(positions_source)

# Win/podium plot
generate_win_plot = time_decorator(generate_win_plot)
win_plot = generate_win_plot(positions_source, driver_id)

# Starting position vs finish position scatter
generate_spvfp_scatter = time_decorator(generate_spvfp_scatter)
spvfp_scatter = generate_spvfp_scatter(driver_results, driver_races, driver_driver_standings)

# Mean lap time rank vs finish position scatter plot
generate_mltr_fp_scatter = time_decorator(generate_mltr_fp_scatter)
mltr_fp_scatter = generate_mltr_fp_scatter(driver_results, driver_races, driver_driver_standings, driver_id)

# Team performance graph and table
generate_team_performance_layout = time_decorator(generate_team_performance_layout)
generate_team_performance_layout(driver_races, positions_source, driver_results)

n = drivers.index.unique().shape[0]
i = 1

for driver_id in drivers.index.unique():
    try:
        name = get_driver_name(driver_id, include_flag=False)
        print(f"Testing driver ID {driver_id}, {name}, {i} / {n}")
        i += 1
        start = time.time()
        get_layout(driver_id=driver_id)
        end = time.time()
        elapsed = 1000 * (end - start)
        print(f"Completed in  {elapsed} milliseconds")
        times["did"].append(driver_id)
        times["time"].append(elapsed)
    except Exception as e:
        print(f"Encountered exception: {e}")
        track = traceback.format_exc()
        print("The traceback is:")
        print(track)
        error_ids.append(driver_id)
    print("=======================================")

print("The following driver IDs had errors: ")
print(error_ids)
print("Times:")
print(times)
times = pd.DataFrame.from_dict(times)
print(times["time"].describe())
print("Outliers:")
print(times[times["time"] > times["time"].std() + times["time"].mean()])
times.to_csv("src/unittests/times.csv")
print("Saved to src/unittests/times.csv")
