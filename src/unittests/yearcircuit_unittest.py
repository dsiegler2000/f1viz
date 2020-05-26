import logging
import traceback
from data_loading.data_loader import load_races, load_lap_times, load_results, load_qualifying, load_pit_stops, \
    load_driver_standings, load_constructor_standings, load_fastest_lap_data
from mode.yearcircuit import get_layout, generate_gap_plot, generate_position_plot, generate_lap_time_plot, \
    generate_pit_stop_plot, detect_mark_safety_car_end, mark_fastest_lap, generate_stats_layout
import time
import pandas as pd

# Biggest JOKE of a "unit test" but it works
from utils import time_decorator

races = load_races()
error_ids = []
times = {
    "rid": [],
    "time": []
}

logger = logging.getLogger()
logger.disabled = True

lap_times = load_lap_times()
results = load_results()
pit_stops = load_pit_stops()
qualifying = load_qualifying()
driver_standings = load_driver_standings()
constructor_standings = load_constructor_standings()
fastest_lap_data = load_fastest_lap_data()

circuit_id = 6
year_id = 2001
race = races[(races["circuitId"] == circuit_id) & (races["year"] == year_id)]

race_id = race.index.values[0]
race_laps = lap_times[lap_times["raceId"] == race_id]
race = races[races.index == race_id]
race_results = results[results["raceId"] == race_id]
race_pit_stops = pit_stops[pit_stops["raceId"] == race_id]
race_quali = qualifying[qualifying["raceId"] == race_id]
race_driver_standings = driver_standings[driver_standings["raceId"] == race_id]
race_constructor_standings = constructor_standings[constructor_standings["raceId"] == race_id]
race_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"] == race_id]

generate_gap_plot = time_decorator(generate_gap_plot)
gap_plot, cached_driver_map = generate_gap_plot(race_laps, race_results)

generate_position_plot = time_decorator(generate_position_plot)
position_plot_linked = generate_position_plot(race_laps, cached_driver_map, linking_plot=gap_plot)
position_plot_unlinked = generate_position_plot(race_laps, cached_driver_map)

generate_lap_time_plot = time_decorator(generate_lap_time_plot)
lap_time_plot_linked_layout, lap_time_plot_linked = generate_lap_time_plot(race_laps, cached_driver_map,
                                                                           linking_plot=gap_plot)
lap_time_plot_unlinked_layout, lap_time_plot_unlinked = generate_lap_time_plot(race_laps, cached_driver_map)

generate_pit_stop_plot = time_decorator(generate_pit_stop_plot)
pit_stop_plot = generate_pit_stop_plot(race_pit_stops, cached_driver_map, race_laps)

all_plots = [gap_plot,
             position_plot_linked, position_plot_unlinked,
             lap_time_plot_linked, lap_time_plot_unlinked,
             pit_stop_plot]

# Mark safety car and fastest lap
detect_mark_safety_car_end = time_decorator(detect_mark_safety_car_end)
mark_fastest_lap = time_decorator(mark_fastest_lap)
sc_disclaimer_div = detect_mark_safety_car_end(race_laps, race, race_results, all_plots)
mark_fastest_lap(race_results, [lap_time_plot_linked, lap_time_plot_unlinked])

# Generate race stats
generate_race_stats_layout = time_decorator(generate_stats_layout)
race_stats_layout = generate_race_stats_layout(race_quali, race_results, race_laps, circuit_id,
                                               race_driver_standings, race_constructor_standings,
                                               race_fastest_lap_data, race_id)

n = races.shape[0]

for race_id in races.index.unique():
    try:
        print(f"Testing race ID {race_id} / {n}")
        year = races.loc[race_id, "year"]
        circuit = races.loc[race_id, "circuitId"]
        start = time.time()
        get_layout(year_id=year, circuit_id=circuit)
        end = time.time()
        elapsed = 1000 * (end - start)
        print(f"Completed in  {elapsed} milliseconds")
        times["rid"].append(race_id)
        times["time"].append(elapsed)
    except Exception as e:
        print(f"Encountered exception: {e}")
        track = traceback.format_exc()
        print("The traceback is:")
        print(track)
        error_ids.append(race_id)
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
