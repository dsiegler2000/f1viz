import logging

from bokeh.layouts import column, row
from bokeh.models import Div, Spacer
from data_loading.data_loader import load_races, load_results, load_driver_standings, load_fastest_lap_data, \
    load_lap_times, load_circuits
from mode import circuitdriver, driverconstructor
from utils import get_circuit_name, get_driver_name, get_constructor_name, plot_image_url, PLOT_BACKGROUND_COLOR

races = load_races()
results = load_results()
driver_standings = load_driver_standings()
fastest_lap_data = load_fastest_lap_data()
lap_times = load_lap_times()
circuits = load_circuits()


def get_layout(circuit_id=-1, driver_id=-1, constructor_id=-1, download_image=True, **kwargs):
    circuit_rids = races[races["circuitId"] == circuit_id].index.values
    driver_results = results[results["driverId"] == driver_id]
    cdc_results = driver_results[(driver_results["raceId"].isin(circuit_rids)) &
                                 (driver_results["constructorId"] == constructor_id)]

    logging.info(f"Generating layout for mode CIRCUITDRIVERCONSTRUCTOR in circuitdriverconstructor, "
                 f"circuit_id={circuit_id}, driver_id={driver_id}, constructor_id={constructor_id}")

    # Detect invalid combos
    if cdc_results.shape[0] == 0:
        return generate_error_layout(circuit_id, driver_id, constructor_id)

    cdc_rids = cdc_results["raceId"]
    cdc_races = races.loc[cdc_rids]
    cdc_years = cdc_races["year"].unique()
    driver_driver_standings = driver_standings[driver_standings["driverId"] == driver_id]
    circuit_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"].isin(circuit_rids)]
    cdc_fastest_lap_data = circuit_fastest_lap_data[(circuit_fastest_lap_data["driver_id"] == driver_id) &
                                                    (circuit_fastest_lap_data["raceId"].isin(cdc_rids))]
    cdc_lap_times = lap_times[(lap_times["raceId"].isin(cdc_rids)) & (lap_times["driverId"] == driver_id)]
    circuit_results = results[results["raceId"].isin(circuit_rids)]
    constructor_results_idxs = []
    for idx, results_row in cdc_results.iterrows():
        constructor_id = results_row["constructorId"]
        rid = results_row["raceId"]
        results_slice = circuit_results[(circuit_results["constructorId"] == constructor_id) &
                                        (circuit_results["raceId"] == rid)]
        constructor_results_idxs.extend(results_slice.index.values.tolist())
    constructor_results = circuit_results.loc[constructor_results_idxs]

    # Positions plot
    positions_plot, positions_source = circuitdriver.generate_positions_plot(cdc_years, driver_driver_standings,
                                                                             driver_results, cdc_fastest_lap_data,
                                                                             cdc_races, driver_id)

    # Win plot
    win_plot = circuitdriver.generate_win_plot(positions_source)

    # Lap time distribution plot
    lap_time_dist = circuitdriver.generate_lap_time_plot(cdc_lap_times, cdc_rids, circuit_id, driver_id)

    # Starting position vs finish position scatter
    spvfp_scatter = circuitdriver.generate_spvfp_scatter(cdc_results, cdc_races, driver_driver_standings)

    # Mean lap time rank vs finish position scatter plot
    mltr_fp_scatter = circuitdriver.generate_mltr_fp_scatter(cdc_results, cdc_races, driver_driver_standings)

    # Finish position distribution plot
    finish_pos_dist = circuitdriver.generate_finishing_position_bar_plot(cdc_results)

    # Teammate comparison line plot
    teammate_comparison_line_plot = generate_teammate_comparison_line_plot(positions_source,
                                                                           constructor_results, driver_id)

    # Results table
    results_table, results_source = circuitdriver.generate_results_table(cdc_results, cdc_fastest_lap_data,
                                                                         circuit_results, circuit_fastest_lap_data)

    # Stats div
    stats_div = circuitdriver.generate_stats_layout(cdc_years, cdc_races, cdc_results, cdc_fastest_lap_data,
                                                    positions_source, circuit_id, driver_id)

    if download_image:
        # Track image
        circuit_row = circuits.loc[circuit_id]
        image_url = str(circuit_row["imgUrl"])
        image_view = plot_image_url(image_url)
        disclaimer = Div(text="The image is of the current configuration of the track.")
        image_view = column([image_view, disclaimer], sizing_mode="stretch_both")
    else:
        image_view = Div()

    header = Div(text=f"<h2><b>{get_driver_name(driver_id)} at {get_circuit_name(circuit_id)} with "
                      f"{get_constructor_name(constructor_id)}</b></h2><br>")

    middle_spacer = Spacer(width=5, background=PLOT_BACKGROUND_COLOR)
    layout = column([header,
                     positions_plot, middle_spacer,
                     win_plot, middle_spacer,
                     lap_time_dist, middle_spacer,
                     row([spvfp_scatter, mltr_fp_scatter], sizing_mode="stretch_width"),
                     finish_pos_dist,
                     teammate_comparison_line_plot,
                     row([image_view], sizing_mode="stretch_width"),
                     results_table,
                     stats_div],
                    sizing_mode="stretch_width")
    logging.info("Finished generating layout for mode CIRCUITDRIVERCONSTRUCTOR")

    return layout


def generate_teammate_comparison_line_plot(positions_source, constructor_results, driver_id):
    """
    Driver finish pos and teammate finish pos vs time.
    :param positions_source: Positions source
    :param constructor_results: Constructor results
    :param driver_id: Driver ID
    :return: Teammate comparison line plot layout
    """
    kwargs = dict(
        return_components_and_source=True,
        default_alpha=0.5,
        mute_smoothed=True
    )
    slider, teammate_fp_plot, source = driverconstructor.generate_teammate_comparison_line_plot(positions_source,
                                                                                                constructor_results,
                                                                                                driver_id,
                                                                                                **kwargs)
    driverconstructor.mark_teammate_changes(positions_source, constructor_results, driver_id, teammate_fp_plot)

    return column([slider, teammate_fp_plot], sizing_mode="stretch_width")


def generate_error_layout(circuit_id, driver_id, constructor_id):
    """
    Generates error layout in the case where the given constructor with the given driver hasn't competed at the given
    circuit.
    :param circuit_id: Circuit ID
    :param driver_id: Driver ID
    :param constructor_id: Constructor ID
    :return: Error layout
    """
    logging.info("Generating error layout")
    circuit_name = get_circuit_name(circuit_id)
    driver_name = get_driver_name(driver_id)
    constructor_name = get_constructor_name(constructor_id)
    circuit_rids = races[races["circuitId"] == circuit_id].index.values
    circuit_results = results[results["raceId"].isin(circuit_rids)]
    # Constructors at this circuit
    constructors_at_circuit = circuit_results["constructorId"].unique()
    # Constructors of this driver at this circuit
    driver_constructors = circuit_results[circuit_results["driverId"] == driver_id]["constructorId"].unique()
    # Drivers for this constructor
    constructor_drivers = circuit_results[circuit_results["constructorId"] == constructor_id]["driverId"].unique()

    # Generate the text
    text = f"Unfortunately, {driver_name} never competed at {circuit_name} with {constructor_name}. " \
           f"Here are some other options:<br>"
    text += f"The following constructors competed at {circuit_name}..."
    text += "<ul>"
    for cid in constructors_at_circuit:
        text += f"<li>{get_constructor_name(cid)}</li>"
    text += "</ul><br>"
    text += f"{driver_name} raced at {circuit_name} while competing with the following constructors..."
    text += "<ul>"
    for cid in driver_constructors:
        text += f"<li>{get_constructor_name(cid)}</li>"
    text += "</ul><br>"
    text += f"{constructor_name} had the following drivers while competing at {circuit_name}..."
    text += "<ul>"
    for did in constructor_drivers:
        text += f"<li>{get_driver_name(did)}</li>"
    text += "</ul><br>"

    layout = Div(text=text)

    return layout
