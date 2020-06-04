import logging
import traceback
import time
import pandas as pd
from data_loading.data_loader import load_drivers, load_results, load_driver_standings, load_races, \
    load_fastest_lap_data
from mode.yeardriver import generate_wdc_plot, generate_positions_plot, get_layout, is_valid_input, \
    generate_teammate_comparison_line_plot, generate_mltr_fp_scatter, generate_spvfp_scatter, \
    generate_finishing_position_bar_plot, generate_win_plot, generate_results_table, generate_stats_layout
from utils import time_decorator, get_driver_name

# Biggest JOKE of a "unit test" but it works

error_ids = []
times = {
    "id": [],
    "time": []
}

logger = logging.getLogger()
logger.disabled = True

drivers = load_drivers()
driver_standings = load_driver_standings()
results = load_results()
races = load_races()
fastest_lap_data = load_fastest_lap_data()

year_id = 2013
driver_id = 20  # Vettel in 2013

year_races = races[races["year"] == year_id]
year_rids = sorted(year_races.index.values)
year_results = results[results["raceId"].isin(year_rids)].sort_values(by="raceId")
year_driver_standings = driver_standings[driver_standings["raceId"].isin(year_rids)].sort_values(by="raceId")
yd_driver_standings = year_driver_standings[year_driver_standings["driverId"] == driver_id]
yd_results = year_results[year_results["driverId"] == driver_id]
year_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"].isin(year_rids)]
yd_fastest_lap_data = year_fastest_lap_data[year_fastest_lap_data["driver_id"] == driver_id]
yd_races = year_races
constructor_results_idxs = []
for idx, results_row in yd_results.iterrows():
    cid = results_row["constructorId"]
    rid = results_row["raceId"]
    constructor_results_idxs.extend(year_results[(year_results["raceId"] == rid) &
                                                 (year_results["constructorId"] == cid)].index.values.tolist())
constructor_results = year_results.loc[constructor_results_idxs]

# More focused WDC plot
generate_wdc_plot = time_decorator(generate_wdc_plot)
wdc_plot = generate_wdc_plot(year_races, year_driver_standings, year_results, driver_id)

# Positions plot
generate_positions_plot = time_decorator(generate_positions_plot)
positions_plot, positions_source = generate_positions_plot(yd_driver_standings, yd_results, yd_fastest_lap_data,
                                                           year_id, driver_id)

# Win plot
generate_win_plot = time_decorator(generate_win_plot)
win_plot = generate_win_plot(positions_source, year_results)

# Finishing position bar plot
generate_finishing_position_bar_plot = time_decorator(generate_finishing_position_bar_plot)
finishing_position_bar_plot = generate_finishing_position_bar_plot(yd_results)

# Start pos vs finish pos scatter
generate_spvfp_scatter = time_decorator(generate_spvfp_scatter)
spvfp_scatter = generate_spvfp_scatter(yd_results, yd_races, yd_driver_standings)

# Mean lap time rank vs finish pos scatter
generate_mltr_fp_scatter = time_decorator(generate_mltr_fp_scatter)
mltr_fp_scatter = generate_mltr_fp_scatter(yd_results, yd_races, yd_driver_standings)

# Teammate comparison line plot
generate_teammate_comparison_line_plot = time_decorator(generate_teammate_comparison_line_plot)
teammate_comparison_line_plot, comparison_source = generate_teammate_comparison_line_plot(positions_source,
                                                                                          constructor_results,
                                                                                          yd_results, driver_id)

# Results table
generate_results_table = time_decorator(generate_results_table)
results_table = generate_results_table(yd_results, yd_fastest_lap_data, year_results, year_fastest_lap_data,
                                       driver_id)

# Stats
generate_stats_layout = time_decorator(generate_stats_layout)
stats_layout = generate_stats_layout(positions_source, comparison_source, constructor_results, year_id, driver_id)

i = 0
invalid = 0
id_list = []
for year_id in races["year"].unique():
    for driver_id in drivers.index.unique():
        id_list.append((year_id, driver_id))
n = len(id_list)
for year_id, driver_id in id_list:
    try:
        i += 1
        if not is_valid_input(year_id, driver_id):
            invalid += 1
            if invalid % 100 == 0:
                print(str(round(100 * invalid / i, 1)) + "% of combos invalid currently")
            continue
        driver_name = get_driver_name(driver_id, include_flag=False, just_last=True)
        print(f"Testing year ID {year_id}, driver ID {driver_id}, {i} / {n}")
        start = time.time()
        get_layout(year_id=year_id, driver_id=driver_id, download_image=False)
        end = time.time()
        elapsed = 1000 * (end - start)
        print(f"Completed in  {elapsed} milliseconds")
        times["id"].append((year_id, driver_id))
        times["time"].append(elapsed)
    except Exception as e:
        print(f"Encountered exception: {e}")
        trace = traceback.format_exc()
        print("The traceback is:")
        print(trace)
        error_ids.append((year_id, driver_id))
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
