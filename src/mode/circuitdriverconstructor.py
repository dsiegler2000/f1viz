import logging
from bokeh.layouts import column
from bokeh.models import Div
from data_loading.data_loader import load_races, load_results, load_driver_standings, load_fastest_lap_data, \
    load_lap_times, load_circuits
from mode import circuitdriver, driverconstructor
from utils import get_circuit_name, get_driver_name, get_constructor_name, plot_image_url, generate_spacer_item, \
    generate_plot_list_selector, generate_div_item, PlotItem, COMMON_PLOT_DESCRIPTIONS

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

    positions_plot, positions_source = circuitdriver.generate_positions_plot(cdc_years, driver_driver_standings,
                                                                             driver_results, cdc_fastest_lap_data,
                                                                             cdc_races, driver_id)
    positions_plot = PlotItem(positions_plot, [], COMMON_PLOT_DESCRIPTIONS["generate_positions_plot"])

    win_plot = PlotItem(circuitdriver.generate_win_plot, [positions_source],
                        COMMON_PLOT_DESCRIPTIONS["generate_win_plot"])

    lap_time_dist = PlotItem(circuitdriver.generate_lap_time_plot, [cdc_lap_times, cdc_rids, circuit_id, driver_id],
                             COMMON_PLOT_DESCRIPTIONS["generate_times_plot"])

    spvfp_scatter = PlotItem(circuitdriver.generate_spvfp_scatter, [cdc_results, cdc_races, driver_driver_standings],
                             COMMON_PLOT_DESCRIPTIONS["generate_spvfp_scatter"])

    mltr_fp_scatter = PlotItem(circuitdriver.generate_mltr_fp_scatter, [cdc_results, cdc_races,
                                                                        driver_driver_standings],
                               COMMON_PLOT_DESCRIPTIONS["generate_mltr_fp_scatter"])

    finish_pos_dist = PlotItem(circuitdriver.generate_finishing_position_bar_plot, [cdc_results],
                               COMMON_PLOT_DESCRIPTIONS["generate_finishing_position_bar_plot"])

    teammate_comparison_line_plot = PlotItem(generate_teammate_comparison_line_plot, [positions_source,
                                                                                      constructor_results, driver_id],
                                             COMMON_PLOT_DESCRIPTIONS["generate_teammate_comparison_line_plot"])

    description = u"Results Table \u2014 table containing results for every time this driver raced at this circuit " \
                  u"with this constructor"
    results_table = PlotItem(circuitdriver.generate_results_table, [cdc_results, cdc_fastest_lap_data, circuit_results,
                                                                    circuit_fastest_lap_data], description)

    stats_div = PlotItem(circuitdriver.generate_stats_layout, [cdc_years, cdc_races, cdc_results, cdc_fastest_lap_data,
                                                               positions_source, circuit_id, driver_id],
                         "description " * 5)

    if download_image:
        # Track image
        circuit_row = circuits.loc[circuit_id]
        image_url = str(circuit_row["imgUrl"])
        image_view = plot_image_url(image_url)
        disclaimer = Div(text="The image is of the current configuration of the track.")
        image_view = column([image_view, disclaimer], sizing_mode="stretch_both")
    else:
        image_view = Div()
    image_view = PlotItem(image_view, [], "", listed=False)

    header = generate_div_item(f"<h2><b>{get_driver_name(driver_id)} at {get_circuit_name(circuit_id)} with "
                               f"{get_constructor_name(constructor_id)}</b></h2><br>")

    middle_spacer = generate_spacer_item()
    layout = generate_plot_list_selector([
        [header],
        [positions_plot], [middle_spacer],
        [win_plot], [middle_spacer],
        [lap_time_dist], [middle_spacer],
        [spvfp_scatter, mltr_fp_scatter], [middle_spacer],
        [finish_pos_dist],
        [teammate_comparison_line_plot],
        [image_view],
        [results_table],
        [stats_div]
    ])

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
