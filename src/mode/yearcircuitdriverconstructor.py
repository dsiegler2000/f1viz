import logging

from bokeh.layouts import column
from bokeh.models import Div, Spacer
from data_loading.data_loader import load_races, load_results, load_lap_times, load_pit_stops, load_qualifying, \
    load_fastest_lap_data, load_driver_standings
from mode import yearcircuitdriver
from utils import get_driver_name, get_circuit_name, get_constructor_name, get_race_name, PLOT_BACKGROUND_COLOR

# ycdc = yearcircuitdriverconstructor

races = load_races()
results = load_results()
lap_times = load_lap_times()
pit_stop_data = load_pit_stops()
quali = load_qualifying()
fastest_lap_data = load_fastest_lap_data()
driver_standings = load_driver_standings()


def get_layout(year_id=-1, circuit_id=-1, driver_id=-1, constructor_id=-1, download_image=True, **kwargs):
    logging.info(f"Generating layout for mode YEARCIRCUITDRIVERCONSTRUCTOR in yearcircuitdriverconstructor, "
                 f"year_id={year_id}, circuit_id={circuit_id}, driver_id={driver_id}, constructor_id={constructor_id}")
    year_races = races[races["year"] == year_id]
    race = year_races[year_races["circuitId"] == circuit_id]
    if race.shape[0] == 0:
        return generate_error_layout(year_id, circuit_id, driver_id, constructor_id)
    rid = race.index.values[0]
    race_results = results[results["raceId"] == rid]
    ycdc_results = race_results[(race_results["driverId"] == driver_id) &
                                (race_results["constructorId"] == constructor_id)]
    if ycdc_results.shape[0] == 0:
        return generate_error_layout(year_id, circuit_id, driver_id, constructor_id)

    race_laps = lap_times[lap_times["raceId"] == rid]
    ycdc_laps = race_laps[race_laps["driverId"] == driver_id]
    ycdc_pit_stop_data = pit_stop_data[(pit_stop_data["raceId"] == rid) & (pit_stop_data["driverId"] == driver_id)]
    race_quali = quali[quali["raceId"] == rid]
    ycdc_fastest_lap_data = fastest_lap_data[(fastest_lap_data["raceId"] == rid) &
                                             (fastest_lap_data["driver_id"] == driver_id)]
    year_driver_standings = driver_standings[driver_standings["raceId"].isin(year_races.index.values)]

    # Gap plot
    gap_plot, cached_driver_map = generate_gap_plot(race_laps, race_results, driver_id)

    # Position plot
    position_plot = generate_position_plot(race_laps, race_results, cached_driver_map, driver_id)

    # Lap time plot
    lap_time_plot = generate_lap_time_plot(race_laps, race_results, cached_driver_map, driver_id)

    plots = [gap_plot, position_plot, lap_time_plot]

    # Mark safety car
    # todo fix
    disclaimer_sc = detect_mark_safety_car(race_laps, race, race_results, plots)

    # Mark fastest lap
    mark_fastest_lap(ycdc_results, plots)

    plots = [gap_plot, lap_time_plot]

    # Mark overtakes
    disclaimer_overtakes = detect_mark_overtakes(ycdc_laps, race_laps, plots)

    # Mark pit stops
    mark_pit_stops(ycdc_pit_stop_data, [gap_plot, lap_time_plot], driver_id)

    # Quali table
    quali_table, quali_source = generate_quali_table(race_quali, race_results, driver_id)

    # Stats layout
    stats_layout = generate_stats_layout(ycdc_results, ycdc_pit_stop_data, ycdc_fastest_lap_data,
                                         year_driver_standings,
                                         race_results, quali_source, rid, circuit_id, driver_id,
                                         download_image=download_image)

    driver_name = get_driver_name(driver_id)
    race_name = get_race_name(rid, include_year=True)
    constructor_name = get_constructor_name(constructor_id)
    header = Div(text=f"<h2><b>What did {driver_name}'s {race_name} with {constructor_name} look like?</b></h2><br><i>"
                      f"Yellow dashed vertical lines show the start of a safety car period, orange vertical lines show "
                      f"the end*. "
                      f"<br>The white line marks the fastest lap of the race."
                      f"<br>Green lines show overtakes and red lines show being overtaken**."
                      f"<br>Pink lines show pit stops along with how long was spent in the pits.</i>")

    middle_spacer = Spacer(width=5, background=PLOT_BACKGROUND_COLOR)
    layout = column([header,
                     gap_plot, middle_spacer,
                     position_plot, middle_spacer,
                     lap_time_plot, middle_spacer,
                     disclaimer_sc,
                     disclaimer_overtakes,
                     quali_table,
                     stats_layout],
                    sizing_mode="stretch_width")

    logging.info("Finished generating layout for mode YEARCIRCUITDRIVERCONSTRUCTOR")

    return layout


def generate_gap_plot(race_laps, race_results, driver_id):
    """
    Generates plot showing gap to leader.
    :param race_laps: Race laps
    :param race_results: Race results
    :param driver_id: Driver ID
    :return: Gap plot layout, cached driver map
    """
    return yearcircuitdriver.generate_gap_plot(race_laps, race_results, driver_id)


def generate_position_plot(race_laps, race_results, cached_driver_map, driver_id):
    """
    Generates position plot.
    :param race_laps: Race laps
    :param race_results: Race results
    :param cached_driver_map: Cached driver map
    :param driver_id: Driver ID
    :return: Position plot layout
    """
    return yearcircuitdriver.generate_position_plot(race_laps, race_results, cached_driver_map, driver_id)


def detect_mark_safety_car(race_laps, race, race_results, plots):
    """
    Detects and marks safety cars
    :param race_laps: Race laps
    :param race: Race (slice of races)
    :param race_results: Race results
    :param plots: Plots to mark
    :return: Disclaimer div
    """
    # todo fix
    return yearcircuitdriver.detect_mark_safety_car(race_laps, race, race_results, plots)


def generate_lap_time_plot(race_laps, race_results, cached_driver_map, driver_id):
    """
    Lap time vs lap plot.
    :param race_laps: Race laps
    :param race_results: Race results
    :param cached_driver_map: Cached driver map
    :param driver_id: Driver ID
    :return: Lap time plot layout
    """
    return yearcircuitdriver.generate_lap_time_plot(race_laps, race_results, cached_driver_map, driver_id)


def mark_fastest_lap(ycdc_results, plots):
    """
    Marks the fastest lap
    :param ycdc_results: YCDC results
    :param plots: Plots to mark
    :return: None
    """
    yearcircuitdriver.mark_fastest_lap(ycdc_results, plots)


def detect_mark_overtakes(ycdc_laps, race_laps, plots):
    """
    Marks overtakes
    :param ycdc_laps: YCDC laps
    :param race_laps: Race laps
    :param plots: Plots to mark
    :return: Disclaimer div
    """
    return yearcircuitdriver.detect_mark_overtakes(ycdc_laps, race_laps, plots)


def mark_pit_stops(ycdc_pit_stop_data, plots, driver_id):
    """
    Marks pit stops
    :param ycdc_pit_stop_data: YCDC pit stop data
    :param plots: Plots to mark
    :param driver_id: Driver ID
    :return: Line dict, which maps driver ID to a list of pit stop lines for use on the legend
    """
    return yearcircuitdriver.mark_pit_stops(ycdc_pit_stop_data, plots, driver_id)


def generate_quali_table(race_quali, race_results, driver_id):
    """
    Generates a table of qualifying results
    :param race_quali: Race quali slice
    :param race_results: Race results
    :param driver_id: Driver ID
    :return: Quali table layout, quali source
    """
    return yearcircuitdriver.generate_quali_table(race_quali, race_results, driver_id)


def generate_stats_layout(ycdc_results, ycdc_pit_stop_data, ycdc_fastest_lap_data, year_driver_standings, race_results,
                          quali_source, rid, circuit_id, driver_id, download_image=True):
    """
    Generates a stats layout
    :param ycdc_results: YCDC results
    :param ycdc_pit_stop_data: YCDC pit stop data
    :param ycdc_fastest_lap_data: YCDC fastest lap data
    :param year_driver_standings: Year driver standings
    :param race_results: Race results
    :param quali_source: Quali source
    :param rid: Race ID
    :param circuit_id: Circuit ID
    :param driver_id: Driver ID
    :param download_image: Whether to actually download the image
    :return: Stats layout
    """
    return yearcircuitdriver.generate_stats_layout(ycdc_results, ycdc_pit_stop_data, ycdc_fastest_lap_data,
                                                   year_driver_standings, race_results, quali_source, rid, circuit_id,
                                                   driver_id, download_image=download_image)


def generate_error_layout(year_id, circuit_id, driver_id, constructor_id):
    logging.info("Generating error layout")
    driver_name = get_driver_name(driver_id)
    circuit_name = get_circuit_name(circuit_id)
    constructor_name = get_constructor_name(constructor_id)

    year_races = races[races["year"] == year_id]
    year_results = results[results["raceId"].isin(year_races.index)]
    constructors_this_driver = year_results[year_results["driverId"] == driver_id]["constructorId"].unique()
    circuit_years = sorted(races[races["circuitId"] == circuit_id]["year"].unique(), reverse=True)

    text = f"Unfortunately, {driver_name} did not compete at {circuit_name} in {year_id} for {constructor_name}.<br>"

    text += f"{driver_name} drove for the following constructors in {year_id}:<br>"
    text += "<ul>"
    for cid in constructors_this_driver:
        text += f"<li>{get_constructor_name(cid)}</li>"
    text += "</ul><br>"
    text += f"The following races happened in {year_id}:<br>"
    text += "<ul>"
    for rid in year_races.index.values:
        text += f"<li>{get_race_name(rid)}</li>"
    text += "</ul><br>"
    text += f"{circuit_name} hosted a Grand Prix in the following years:<br>"
    text += "<ul>"
    for year_id in circuit_years:
        text += f"<li>{year_id}</li>"
    text += "</ul><br>"
    layout = Div(text=text)
    return layout

