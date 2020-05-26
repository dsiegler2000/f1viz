import logging
import traceback
import time
import pandas as pd
from data_loading.data_loader import load_races, load_fastest_lap_data, load_results, load_constructors, \
    load_constructor_standings
from mode.constructor import generate_positions_plot, get_layout, generate_circuit_performance_table, \
    generate_finishing_positions_bar_plot, generate_wcc_position_bar_plot, generate_win_plot, \
    generate_driver_performance_table, generate_stats_layout
from utils import time_decorator, get_constructor_name

# Biggest JOKE of a "unit test" but it works
error_ids = []
times = {
    "cid": [],
    "time": []
}

logger = logging.getLogger()
logger.disabled = True

constructors = load_constructors()
results = load_results()
constructor_standings = load_constructor_standings()
races = load_races()
fastest_lap_data = load_fastest_lap_data()
constructor_id = 6  # Ferrari

constructor_results = results[results["constructorId"] == constructor_id]
constructor_constructor_standings = constructor_standings[constructor_standings["constructorId"] == constructor_id]
constructor_rids = constructor_results["raceId"].unique()
constructor_races = races[races.index.isin(constructor_rids)].sort_values(by=["year", "raceId"])
constructor_years = constructor_races[constructor_races.index.isin(constructor_constructor_standings["raceId"])]
constructor_years = constructor_years["year"].unique()
constructor_fastest_lap_data = fastest_lap_data[fastest_lap_data["constructor_id"] == constructor_id]

# Position plot
generate_positions_plot = time_decorator(generate_positions_plot)
positions_plot, positions_source = generate_positions_plot(constructor_years, constructor_constructor_standings,
                                                           constructor_results, constructor_fastest_lap_data,
                                                           constructor_id)

# Positions bar plot
generate_finishing_positions_bar_plot = time_decorator(generate_finishing_positions_bar_plot)
positions_bar_plot = generate_finishing_positions_bar_plot(constructor_results)

# WCC bar plot
generate_wcc_position_bar_plot = time_decorator(generate_wcc_position_bar_plot)
wcc_bar_plot = generate_wcc_position_bar_plot(positions_source)

# Win plot
generate_win_plot = time_decorator(generate_win_plot)
win_plot = generate_win_plot(positions_source, constructor_id)

# Circuit performance table
generate_circuit_performance_table = time_decorator(generate_circuit_performance_table)
circuit_performance_table = generate_circuit_performance_table(constructor_results, constructor_races,
                                                               constructor_id, consider_up_to=24)

# Driver performance graph and table
generate_driver_performance_layout = time_decorator(generate_driver_performance_table)
driver_performance_layout, driver_performance_source = generate_driver_performance_layout(constructor_races,
                                                                                          constructor_results)

# Constructor stats div
generate_constructor_stats_layout = time_decorator(generate_stats_layout)
constructor_stats = generate_constructor_stats_layout(constructor_years, constructor_races,
                                                      driver_performance_source, constructor_results,
                                                      constructor_constructor_standings, constructor_id)

n = constructors.index.unique().shape[0]
i = 1

for constructor_id in constructors.index.unique():
    try:
        name = get_constructor_name(constructor_id, include_flag=False)
        print(f"Testing constructor ID {constructor_id}, {name}, {i} / {n}")
        i += 1
        start = time.time()
        get_layout(constructor_id=constructor_id)
        end = time.time()
        elapsed = 1000 * (end - start)
        print(f"Completed in  {elapsed} milliseconds")
        times["cid"].append(constructor_id)
        times["time"].append(elapsed)
    except Exception as e:
        print(f"Encountered exception: {e}")
        track = traceback.format_exc()
        print("The traceback is:")
        print(track)
        error_ids.append(constructor_id)
    print("=======================================")

print("The following constructors IDs had errors: ")
print(error_ids)
print("Times:")
print(times)
times = pd.DataFrame.from_dict(times)
print(times["time"].describe())
print("Outliers:")
print(times[times["time"] > times["time"].std() + times["time"].mean()])
times.to_csv("src/unittests/times.csv")
print("Saved to src/unittests/times.csv")
