import logging
import traceback
import time
import pandas as pd
from data_loading.data_loader import load_circuits, load_races, load_fastest_lap_data, load_qualifying, load_results
from mode.circuit import get_layout, generate_times_plot, generate_spmfp_plot, generate_circuit_results_table, \
    generate_stats_layout, generate_winners_table
from utils import time_decorator, get_circuit_name

# Biggest JOKE of a "unit test" but it works
circuits = load_circuits()
error_ids = []
times = {
    "cid": [],
    "time": []
}

logger = logging.getLogger()
logger.disabled = True

races = load_races()
fastest_lap_data = load_fastest_lap_data()
qualifying = load_qualifying()
results = load_results()
circuit_id = 14  # Monza

circuit_races = races[races["circuitId"] == circuit_id]
circuit_rids = circuit_races.index
circuit_years = sorted(circuit_races["year"].values.tolist())
circuit_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"].isin(circuit_rids)]
circuit_quali = qualifying[qualifying["raceId"].isin(circuit_rids)]
circuit_results = results[results["raceId"].isin(circuit_rids)]

# Generate times plot
generate_times_plot = time_decorator(generate_times_plot)
times_plot = generate_times_plot(circuit_years, circuit_quali, circuit_fastest_lap_data, circuit_races,
                                 circuit_results, circuit_id)

# Generate starting position minus finish position plot
generate_spmfp_plot = time_decorator(generate_spmfp_plot)
spmfp_plot = generate_spmfp_plot(circuit_years, circuit_races, circuit_results)

# Results table
generate_circuit_results_table = time_decorator(generate_circuit_results_table)
circuit_results_table = generate_circuit_results_table(circuit_years, circuit_races, circuit_results,
                                                       circuit_quali, circuit_fastest_lap_data)

generate_circuit_stats_layout = time_decorator(generate_stats_layout)
circuit_stats = generate_circuit_stats_layout(circuit_id, circuit_years, circuit_fastest_lap_data, circuit_results,
                                              circuit_races)

generate_winners_table = time_decorator(generate_winners_table)
winners_table = generate_winners_table(circuit_years, circuit_results, circuit_races)

n = circuits.index.unique().shape[0]
i = 1

for circuit_id in circuits.index.unique():
    try:
        name = get_circuit_name(circuit_id, include_flag=False)
        print(f"Testing circuit ID {circuit_id}, {name}, {i} / {n}")
        i += 1
        start = time.time()
        get_layout(circuit_id=circuit_id)
        end = time.time()
        elapsed = 1000 * (end - start)
        print(f"Completed in  {elapsed} milliseconds")
        times["cid"].append(circuit_id)
        times["time"].append(elapsed)
    except Exception as e:
        print(f"Encountered exception: {e}")
        track = traceback.format_exc()
        print("The traceback is:")
        print(track)
        error_ids.append(circuit_id)
    print("=======================================")

print("The following circuit IDs had errors: ")
print(error_ids)
print("Times:")
print(times)
times = pd.DataFrame.from_dict(times)
print(times["time"].describe())
print("Outliers:")
print(times[times["time"] > times["time"].std() + times["time"].mean()])
times.to_csv("src/unittests/times.csv")
print("Saved to src/unittests/times.csv")
