import logging
from bokeh.layouts import column
from bokeh.models import Div
from data_loading.data_loader import load_results, load_races, load_fastest_lap_data, load_lap_times, \
    load_driver_standings, load_constructor_standings, load_circuits
from mode import circuitdriver, constructor, yearconstructor, driver
from utils import get_circuit_name, get_constructor_name, plot_image_url, generate_spacer_item, \
    generate_div_item, PlotItem, COMMON_PLOT_DESCRIPTIONS, generate_plot_list_selector

# Note, CC = circuit constructor

results = load_results()
races = load_races()
fastest_lap_data = load_fastest_lap_data()
lap_times = load_lap_times()
driver_standings = load_driver_standings()
constructor_standings = load_constructor_standings()
circuits = load_circuits()


def get_layout(circuit_id=-1, constructor_id=-1, download_image=True, **kwargs):
    # Grab some useful slices
    circuit_rids = races[races["circuitId"] == circuit_id].index.values
    constructor_results = results[results["constructorId"] == constructor_id]
    cc_results = constructor_results[constructor_results["raceId"].isin(circuit_rids)]
    cc_rids = cc_results["raceId"]
    cc_races = races[races.index.isin(cc_rids)]

    logging.info(f"Generating layout for mode CIRCUITCONSTRUCTOR in circuitconstructor, circuit_id={circuit_id}, "
                 f"constructor_id={constructor_id}")

    # Detect invalid combos
    if cc_races.shape[0] == 0:
        return generate_error_layout(circuit_id, constructor_id)

    # Generate some more slices
    circuit_results = results[results["raceId"].isin(circuit_rids)]
    circuit_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"].isin(circuit_rids)]
    cc_years = cc_races["year"].unique()
    cc_years.sort()
    cc_fastest_lap_data_idxs = []
    cc_lap_time_idxs = []
    cc_driver_standings_idxs = []
    for idx, results_row in cc_results.iterrows():
        rid = results_row["raceId"]
        did = results_row["driverId"]
        fl_slice = circuit_fastest_lap_data[circuit_fastest_lap_data["driver_id"] == did]
        cc_fastest_lap_data_idxs.extend(fl_slice.index.values.tolist())
        lt_slice = lap_times[(lap_times["raceId"] == rid) & (lap_times["driverId"] == did)]
        cc_lap_time_idxs.extend(lt_slice.index.values.tolist())
        driver_standings_slice = driver_standings[(driver_standings["raceId"] == rid) &
                                                  (driver_standings["driverId"] == did)]
        cc_driver_standings_idxs.extend(driver_standings_slice.index.values.tolist())
    cc_fastest_lap_data = circuit_fastest_lap_data.loc[cc_fastest_lap_data_idxs]
    cc_lap_times = lap_times.loc[cc_lap_time_idxs]
    cc_driver_standings = driver_standings.loc[cc_driver_standings_idxs]
    constructor_constructor_standings = constructor_standings[constructor_standings["constructorId"] == constructor_id]

    positions_plot, positions_source = generate_positions_plot(cc_years, cc_fastest_lap_data, constructor_results,
                                                               constructor_constructor_standings, cc_races,
                                                               constructor_id)
    positions_plot = PlotItem(positions_plot, [], COMMON_PLOT_DESCRIPTIONS["generate_positions_plot"])

    win_plot = PlotItem(generate_win_plot, [positions_source, constructor_id],
                        COMMON_PLOT_DESCRIPTIONS["generate_win_plot"])

    lap_time_distribution_plot = PlotItem(generate_lap_time_plot, [cc_lap_times, cc_rids, circuit_id, constructor_id],
                                          COMMON_PLOT_DESCRIPTIONS["generate_times_plot"])

    finish_position_bar_plot = PlotItem(generate_finishing_position_bar_plot, [cc_results],
                                        COMMON_PLOT_DESCRIPTIONS["generate_finishing_position_bar_plot"])

    spvfp_scatter = PlotItem(generate_spvfp_scatter, [cc_results, cc_races, cc_driver_standings],
                             COMMON_PLOT_DESCRIPTIONS["generate_spvfp_scatter"])

    mltr_fp_scatter = PlotItem(generate_mltr_fp_scatter, [cc_results, cc_races, cc_driver_standings],
                               COMMON_PLOT_DESCRIPTIONS["generate_mltr_fp_scatter"])

    description = u"Various statistics on this constructor at this circuit"
    stats_div = PlotItem(generate_stats_layout, [cc_years, cc_races, cc_results, cc_fastest_lap_data, positions_source,
                                                 circuit_id, constructor_id], description)

    description = u"Results Table \u2014 table showing this constructor's results at every race this circuit"
    results_table = PlotItem(generate_results_table, [cc_results, cc_fastest_lap_data, circuit_results,
                                                      circuit_fastest_lap_data], description)

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

    circuit_name = get_circuit_name(circuit_id)
    constructor_name = get_constructor_name(constructor_id)
    header = generate_div_item(f"<h2><b>What did/does {constructor_name}'s performance at "
                               f"{circuit_name} look like?</b></h2><br>")

    middle_spacer = generate_spacer_item()
    group = generate_plot_list_selector([
        [header],
        [positions_plot], [middle_spacer],
        [win_plot], [middle_spacer],
        [lap_time_distribution_plot], [middle_spacer],
        [finish_position_bar_plot], [middle_spacer],
        [spvfp_scatter, mltr_fp_scatter], [middle_spacer],
        [image_view],
        [stats_div],
        [results_table]
    ])

    logging.info("Finished generating layout for mode CIRCUITCONSTRUCTOR")

    return group


def generate_lap_time_plot(cc_lap_times, cc_rids, circuit_id, constructor_id):
    """
    Plot this constructor's distribution of lap times compared to all constructors' distribution of lap times for those
    time ranges to show how fast and consistent they are (uses method from `circuitdriver`).
    :param cc_lap_times: CC lap times
    :param cc_rids: CC race ID
    :param circuit_id: Circuit ID
    :param constructor_id: Constructor ID
    :return: Lap time distribution plot layout
    """
    return circuitdriver.generate_lap_time_plot(cc_lap_times, cc_rids, circuit_id, None,
                                                constructor_id=constructor_id)


def generate_positions_plot(cc_years, cc_fastest_lap_data, constructor_results, constructor_constructor_standings,
                            cc_races, constructor_id):
    """
    Plot the quali, finishing position, mean finish position that year, and average lap rank vs time to show improvement
    on that circuit and on the same plot show the quali and average race times to show car+driver improvement, along
    with smoothed versions of all of these.
    :param cc_years: CC years
    :param cc_fastest_lap_data: CC fastest lap data
    :param constructor_results: Constructor results (not standings)
    :param constructor_constructor_standings: Constructor constructor standings
    :param cc_races: CC races
    :param constructor_id: Constructor ID
    :return: Positions plot layout, positions source
    """
    return constructor.generate_positions_plot(cc_years, constructor_constructor_standings, constructor_results,
                                               cc_fastest_lap_data, constructor_id, include_lap_times=True,
                                               races_sublist=cc_races, show_mean_finish_pos=True)


def generate_win_plot(positions_source, constructor_id):
    """
    Plot number of races, number of wins, win percent, number of podiums, podium percent, number of DNFs, DNF percent,
    and also number of top 6 finishes vs time all on one graph for this circuit (see `constructor.generate_win_plot`).
    :param positions_source: Positions source
    :param constructor_id: Constructor ID
    :return:
    """
    return constructor.generate_win_plot(positions_source, constructor_id)


def generate_results_table(cc_results, cc_fastest_lap_data, circuit_results, circuit_fastest_lap_data):
    """
    Table of all of the constructor's finishes at that circuit, including place/why they DNF'd, fastest lap time,
    average lap time, and any other info available (see `yearconstructor.generate_results_table` or
    `circuitdriver.generate_results_table`)
    :param cc_results: CC results
    :param cc_fastest_lap_data: CC fastest lap data
    :param circuit_results: Circuit results
    :param circuit_fastest_lap_data: Circuit fastest lap data
    :return: Results table (no source)
    """
    return yearconstructor.generate_results_table(cc_results, cc_fastest_lap_data, circuit_results,
                                                  circuit_fastest_lap_data, year_only=True, height=530)[0]


def generate_finishing_position_bar_plot(cc_results):
    """
    Plot histogram of results at this circuit (should be easy just use method in `driver`).
    :param cc_results: CC results
    :return: Finishing position bar plot layout
    """
    return constructor.generate_finishing_position_bar_plot(cc_results)


def generate_spvfp_scatter(cc_results, cc_races, cc_driver_standings):
    """
    Plot a scatter of quali position vs finish position and draw the y=x line (use existing method).
    :param cc_results: CC results
    :param cc_races: CC races
    :param cc_driver_standings: CC driver standings
    :return: Start pos vs finish pos scatter plot
    """
    return driver.generate_spvfp_scatter(cc_results, cc_races, cc_driver_standings, include_year_labels=True,
                                         color_drivers=True)


def generate_mltr_fp_scatter(cc_results, cc_races, cc_driver_standings):
    """
    Plot scatter of finish position vs mean lap time rank to get a sense of what years the driver out-drove the car.
    Use method in `driver`
    :param cc_results: CC results
    :param cc_races: CC races
    :param cc_driver_standings: CC driver standings
    :return: Mean lap time rank vs finish pos scatter plot
    """
    return driver.generate_mltr_fp_scatter(cc_results, cc_races, cc_driver_standings, include_year_labels=True,
                                           color_drivers=True)


def generate_stats_layout(cc_years, cc_races, cc_results, cc_fastest_lap_data, positions_source,
                          circuit_id, constructor_id):
    """
    Have some type of stats div restating number of races, wins, podiums, best results, track image, weather at each
    race maybe, DNFs, and anything else typically included. See other modes for inspiration.
    Stats div including:
    - Years
    - Num. races
    - Num. wins
    - Num. podiums
    - Best results
    - Average start position
    - Average finish position
    - Average lap time
    - Fastest lap time
    - DNF info
    - Compared to other circuits
    :param cc_years: CC years
    :param cc_races: CC races
    :param cc_results: CC results
    :param cc_fastest_lap_data: CC fastest lap data
    :param positions_source: Positions source
    :param circuit_id: Circuit ID
    :param constructor_id: Constructor ID
    :return: Stats layout
    """
    return circuitdriver.generate_stats_layout(cc_years, cc_races, cc_results, cc_fastest_lap_data, positions_source,
                                               circuit_id, None, constructor_id=constructor_id)


def generate_error_layout(circuit_id, constructor_id):
    """
    Generates an error layout in the event that the given constructor never competed at the given circuit.
    :param circuit_id: Circuit ID
    :param constructor_id: Constructor ID
    :return: Div layout
    """
    logging.info("Generating error layout")
    circuit_name = get_circuit_name(circuit_id, include_flag=False)
    constructor_name = get_constructor_name(constructor_id, include_flag=False)
    circuit_rids = races[races["circuitId"] == circuit_id].index.values
    constructors_at_that_circuit = results[results["raceId"].isin(circuit_rids)]["constructorId"].unique()
    rids_for_this_constructor = results[results["constructorId"] == constructor_id]["raceId"]
    circuits_for_this_constructor = races[races.index.isin(rids_for_this_constructor)]["circuitId"].unique()

    # Generate the text
    text = f"Unfortunately, {constructor_name} never competed at {circuit_name}. Here are some other options:<br>"
    text += f"{constructor_name} competed at the following circuits..."
    text += "<ul>"
    for cid in circuits_for_this_constructor:
        text += f"<li>{get_circuit_name(cid)}</li>"
    text += "</ul>"
    text += f"The {circuit_name} hosted the following constructors..."
    text += "<ul>"
    for cid in constructors_at_that_circuit:
        text += f"<li>{get_constructor_name(cid)}</li>"
    text += "</ul><br>"

    layout = Div(text=text)
    return layout


def is_valid_input(circuit_id, constructor_id):
    """
    Returns True if the given combination of circuit and constructor are valid
    :param circuit_id: Circuit ID
    :param constructor_id: Constructor ID
    :return: True if valid, False otherwise
    """
    circuit_rids = races[races["circuitId"] == circuit_id].index.values
    constructor_results = results[results["constructorId"] == constructor_id]
    cc_results = constructor_results[constructor_results["raceId"].isin(circuit_rids)]
    cc_rids = cc_results["raceId"]
    cc_races = races[races.index.isin(cc_rids)]
    return cc_races.shape[0] > 0
