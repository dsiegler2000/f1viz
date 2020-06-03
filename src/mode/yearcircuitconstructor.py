import logging
from bokeh.layouts import column, row
from bokeh.models import Div, Spacer
from data_loading.data_loader import load_results, load_races, load_lap_times, load_pit_stops, load_qualifying, \
    load_circuits, load_fastest_lap_data, load_driver_standings, load_constructor_standings
from mode import yearcircuit, yearcircuitdriver
from utils import get_constructor_name, get_circuit_name, rounds_to_str, PLOT_BACKGROUND_COLOR, get_race_name, \
    plot_image_url, get_driver_name, int_to_ordinal, result_to_str, millis_to_str, vdivider
import numpy as np

# Note, ycc = yearcircuitconstructor

races = load_races()
results = load_results()
lap_times = load_lap_times()
pit_stop_data = load_pit_stops()
quali = load_qualifying()
circuits = load_circuits()
fastest_lap_data = load_fastest_lap_data()
driver_standings = load_driver_standings()
constructor_standings = load_constructor_standings()


def get_layout(year_id=-1, circuit_id=-1, constructor_id=-1, download_image=True, **kwargs):
    # Generate slices
    year_races = races[races["year"] == year_id]
    race = year_races[year_races["circuitId"] == circuit_id]
    if race.shape[0] == 0:
        return generate_error_layout(year_id=year_id, circuit_id=circuit_id, constructor_id=constructor_id)
    rid = race.index.values[0]
    race_results = results[results["raceId"] == rid]
    ycc_results = race_results[race_results["constructorId"] == constructor_id]

    logging.info(f"Generating layout for mode YEARCIRCUITCONSTRUCTOR in yearcircuitconstructor, year_id={year_id}, "
                 f"circuit_id={circuit_id}, constructor_id={constructor_id}")

    if ycc_results.shape[0] == 0:
        return generate_error_layout(year_id=year_id, circuit_id=circuit_id, constructor_id=constructor_id)

    # Generate more slices
    driver_ids = ycc_results["driverId"].unique()
    ycc_pit_stop_data = pit_stop_data[(pit_stop_data["raceId"] == rid) & (pit_stop_data["driverId"].isin(driver_ids))]
    race_laps = lap_times[lap_times["raceId"] == rid]
    race_quali = quali[quali["raceId"] == rid]
    ycc_fastest_lap_data = fastest_lap_data[(fastest_lap_data["raceId"] == rid) &
                                            (fastest_lap_data["driver_id"].isin(driver_ids))]
    year_driver_standings = driver_standings[driver_standings["raceId"].isin(year_races.index.values)]
    year_constructor_standings = constructor_standings[constructor_standings["raceId"].isin(year_races.index.values)]

    # Gap plot
    gap_plot, cached_driver_map = generate_gap_plot(race_laps, race_results, driver_ids, constructor_id)

    # Position plot
    position_plot = generate_position_plot(race_laps, race_results, cached_driver_map, driver_ids, constructor_id)

    # Lap time plot
    lap_time_plot = generate_lap_time_plot(race_laps, race_results, cached_driver_map, driver_ids, constructor_id)

    plots = [gap_plot, position_plot, lap_time_plot]

    # Mark pit stops
    mark_pit_stops(ycc_pit_stop_data, driver_ids, cached_driver_map, plots)

    # Mark safety car
    disclaimer_sc = detect_mark_safety_car(race_laps, race, race_results, plots)

    # Quali table
    quali_table, quali_source = generate_quali_table(race_quali, race_results, driver_ids)

    # Stats layout
    stats_layout = generate_stats_layout(ycc_results, ycc_pit_stop_data, ycc_fastest_lap_data, year_driver_standings,
                                         year_constructor_standings, quali_source, rid, circuit_id, constructor_id,
                                         driver_ids, download_image=download_image)

    constructor_name = get_constructor_name(constructor_id)
    race_name = get_race_name(rid, include_year=True)
    header = Div(text=f"<h2><b>What did {constructor_name}'s {race_name} look like?</b></h2><br><i>Yellow dashed "
                      f"vertical lines show the start of a safety car period, orange vertical lines show the end.*"
                      f"<br>The white line marks the fastest lap of the race."
                      f"<br>Pink lines show pit stops along with how long was spent in the pits.")

    middle_spacer = Spacer(width=5, background=PLOT_BACKGROUND_COLOR)
    layout = column([header,
                     gap_plot, middle_spacer,
                     position_plot, middle_spacer,
                     lap_time_plot, middle_spacer,
                     disclaimer_sc,
                     quali_table,
                     stats_layout],
                    sizing_mode="stretch_width")

    logging.info("Finished generating layout for mode YEARCIRCUITCONSTRUCTOR")

    return layout


def generate_gap_plot(race_laps, race_results, driver_ids, constructor_id, consider_window=1):
    """
    Plots gap to leader.
    :param race_laps: Race laps
    :param race_results: Race results
    :param driver_ids: Driver IDs of the drivers on the team
    :param constructor_id: Constructor ID
    :param consider_window: Window to focus on (places around the driver)
    :return: Gap plot layout, cached driver map
    """
    muted_dids = generate_muted_dids(race_results, driver_ids, constructor_id, consider_window)
    return yearcircuit.generate_gap_plot(race_laps, race_results, highlight_dids=driver_ids, muted_dids=muted_dids)


def generate_position_plot(race_laps, race_results, cached_driver_map, driver_ids, constructor_id, consider_window=1):
    """
    Generates a position plot (position vs lap)
    :param race_laps: Race laps
    :param race_results: Race results
    :param cached_driver_map: Cached driver map
    :param driver_ids: Driver IDs of the drivers on the team
    :param constructor_id: Constructor ID
    :param consider_window: Consider window
    :return: Position plot
    """
    muted_dids = generate_muted_dids(race_results, driver_ids, constructor_id, consider_window)
    return yearcircuit.generate_position_plot(race_laps, cached_driver_map, highlight_dids=driver_ids,
                                              muted_dids=muted_dids)


def generate_lap_time_plot(race_laps, race_results, cached_driver_map, driver_ids, constructor_id, consider_window=1):
    """
    Generates a plot of lap time vs laps
    :param race_laps: Race laps
    :param race_results: Race results
    :param cached_driver_map: Cached driver map
    :param driver_ids: Driver IDs of the drivers on the team
    :param constructor_id: Constructor ID
    :param consider_window: Window to focus on (places around the driver)
    :return: Lap time plot layout
    """
    muted_dids = generate_muted_dids(race_results, driver_ids, constructor_id, consider_window)
    kwargs = dict(
        stdev_range=(1.7, 0.4),
        include_hist=False,
        highlight_dids=driver_ids,
        muted_dids=muted_dids
    )
    return yearcircuit.generate_lap_time_plot(race_laps, cached_driver_map, **kwargs)[1]


def detect_mark_safety_car(race_laps, race, race_results, plots):
    """
    Detect and mark safety car laps.
    :param race_laps: Race laps
    :param race: Race entry (from races.csv)
    :param race_results: Race results
    :param plots: Plots
    :return: Disclaimer div
    """
    disclaimer = yearcircuit.detect_mark_safety_car_end(race_laps, race, race_results, plots)
    return disclaimer


def mark_pit_stops(ycc_pit_stop_data, driver_ids, cached_driver_map, plots):
    h_pct = 0.4
    dh = 0.3
    for did in driver_ids:
        ycd_pit_stop_data = ycc_pit_stop_data[ycc_pit_stop_data["driverId"] == did]
        yearcircuitdriver.mark_pit_stops(ycd_pit_stop_data, plots, did, cached_driver_map=cached_driver_map,
                                         h_pct=h_pct, show_name=True)
        h_pct += dh


def generate_quali_table(race_quali, race_results, driver_ids):
    """
    Generates qualifying table with driver highlighted.
    :param race_quali: Race qualifying data
    :param race_results: Race results
    :param driver_ids: Driver IDs
    :return: Quali table, quali source
    """
    quali_table, pole_info, quali_source = yearcircuit.generate_quali_table(race_quali, race_results,
                                                                            highlight_dids=driver_ids)
    return quali_table, quali_source


def generate_stats_layout(ycc_results, ycc_pit_stop_data, ycc_fastest_lap_data, year_driver_standings,
                          year_constructor_standings, quali_source, race_id, circuit_id, constructor_id, driver_ids,
                          download_image=True):
    """
    Stats div including:
    - Location
    - Date
    - Weather
    - Rating
    - WCC impact
    - For each driver:
        - Qualifying position and time
        - Laps
        - Fastest lap time along with rank
        - Average lap time
        - Finish position
        - Finish time
        - Points scored
        - Num pit stops
        - WDC impact (WDC place before, WDC after)
    :param ycc_results: YCC results
    :param ycc_pit_stop_data: YCC pit stop data
    :param ycc_fastest_lap_data: YCC fastest lap data
    :param year_driver_standings: Year driver standings
    :param year_constructor_standings: Year constructor standings
    :param quali_source: Quali source
    :param race_id: Race ID
    :param circuit_id: Circuit ID
    :param constructor_id: Constructor ID
    :param driver_ids: Driver IDs
    :param download_image: Whether to download the track image
    :return: Stats layout
    """
    logging.info("Generating race stats layout")
    # Track image
    if download_image:
        image_url = str(circuits.loc[circuit_id, "imgUrl"])
        image_view = plot_image_url(image_url)
        disclaimer = Div(text="The image is of the current configuration of the track.")
        image_view = column([image_view, disclaimer], sizing_mode="stretch_both")
    else:
        image_view = Div()
    # Race info
    race = races.loc[race_id]
    round_num = race["round"]
    circuit = circuits.loc[circuit_id]
    date = race["datetime"].split(" ")
    if len(date) > 0:
        date_str = date[0]
    else:
        date_str = race["datetime"]
    location_str = circuit["location"] + ", " + circuit["country"]
    circuit_str = circuit["name"] + " (" + location_str + ")"
    weather = race["weather"]
    if weather is None or weather == "":
        weather_str = ""
    else:
        weather_str = str(weather).title()
    sc = race["SCLaps"]
    if np.isnan(sc):
        sc_str = ""
    else:
        sc_str = str(int(sc)) + " laps under safety car"
    rating = race["rating"]
    if np.isnan(rating):
        rating_str = ""
    else:
        rating_str = str(round(rating, 1)) + " / 10"
    ycc_constructor_standings = year_constructor_standings[(year_constructor_standings["raceId"] == race_id) &
                                                           (year_constructor_standings["constructorId"] ==
                                                            constructor_id)]
    ycc_constructor_standings_prev = year_constructor_standings[(year_constructor_standings["raceId"] == race_id - 1) &
                                                                (year_constructor_standings["constructorId"] ==
                                                                constructor_id)]
    if round_num == 1 and ycc_constructor_standings.shape[0] > 0:
        wcc_impact_str = int_to_ordinal(ycc_constructor_standings["positionText"].values[0])
    elif round_num > 1 and ycc_constructor_standings.shape[0] > 0 and ycc_constructor_standings_prev.shape[0] > 0:
        wcc_impact_str = "from " + int_to_ordinal(ycc_constructor_standings_prev["position"].values[0]) + " to " + \
                         int_to_ordinal(ycc_constructor_standings["position"].values[0])
    else:
        wcc_impact_str = ""

    header_template = """
    <h2 style="text-align: center;"><b>{}</b></h2>
    """

    subheader_template = """
    <h3 style="text-align: center;"><b>{}</b></h3>
    """

    template = """
    <pre><b>{}</b> {}<br></pre>
    """

    race_name = get_race_name(race_id, include_year=True)
    constructor_name = get_constructor_name(constructor_id)
    ycc_stats = header_template.format(constructor_name + " at the " + race_name)

    ycc_stats += template.format("Circuit Name: ".ljust(22), circuit_str)
    ycc_stats += template.format("Date: ".ljust(22), date_str)
    if weather_str != "" and weather_str.lower() != "nan":
        ycc_stats += template.format("Weather: ".ljust(22), weather_str)
    if not np.isnan(rating):
        ycc_stats += template.format("Rating: ".ljust(22), rating_str)
    if not np.isnan(sc):
        ycc_stats += template.format("Safety Car Laps: ".ljust(22), sc_str)
    if wcc_impact_str != "":
        if round_num == 1:
            ycc_stats += template.format("WCC Position: ".ljust(22), wcc_impact_str)
        else:
            ycc_stats += template.format("WCC Impact: ".ljust(22), wcc_impact_str)
    for driver_id in driver_ids:
        ycd_pit_stop_data = ycc_pit_stop_data[ycc_pit_stop_data["driverId"] == driver_id]
        ycd_results = ycc_results[ycc_results["driverId"] == driver_id]
        ycd_fastest_lap_data = ycc_fastest_lap_data[ycc_fastest_lap_data["driver_id"] == driver_id]
        ycd_driver_standings = year_driver_standings[(year_driver_standings["raceId"] == race_id) &
                                                     (year_driver_standings["driverId"] == driver_id)]
        ycd_driver_standings_prev = year_driver_standings[(year_driver_standings["raceId"] == race_id - 1) &
                                                          (year_driver_standings["driverId"] == driver_id)]
        if round_num == 1 and ycd_driver_standings.shape[0] > 0:
            wdc_impact_str = int_to_ordinal(ycd_driver_standings["positionText"].values[0])
        elif round_num > 1 and ycd_driver_standings.shape[0] > 0 and ycd_driver_standings_prev.shape[0] > 0:
            wdc_impact_str = "from " + int_to_ordinal(ycd_driver_standings_prev["position"].values[0]) + " to " + \
                             int_to_ordinal(ycd_driver_standings["position"].values[0])
        else:
            wdc_impact_str = ""
        if ycd_pit_stop_data.shape[0] > 0:
            num_pit_stops_str = str(ycd_pit_stop_data.shape[0])
        else:
            num_pit_stops_str = ""
        if ycd_results.shape[0] > 0:
            ycd_results_row = ycd_results.iloc[0]
            grid_str = int_to_ordinal(ycd_results_row["grid"])
            fp_str, _ = result_to_str(ycd_results_row["positionOrder"], ycd_results_row["statusId"])
            laps_str = str(ycd_results_row["laps"])
            runtime_str = millis_to_str(ycd_results_row["milliseconds"])
            points = ycd_results_row["points"]
            if abs(int(points) - points) < 0.01:
                points = int(points)
            points_str = str(points)
        else:
            grid_str = ""
            fp_str = ""
            laps_str = ""
            runtime_str = ""
            points_str = ""
        ycd_quali_source = quali_source[quali_source["driver_id"] == driver_id]
        if ycd_quali_source.shape[0] > 0:
            ycd_quali_row = ycd_quali_source.iloc[0]
            quali_pos = ycd_quali_row["quali_position"]
            quali_pos_str = int_to_ordinal(quali_pos)
            quali_time_str = ""
            if "q1" in ycd_quali_source.columns and ycd_quali_row["q1"] != "~":
                quali_time_str = ycd_quali_row["q1"]
            if "q2" in ycd_quali_source.columns and ycd_quali_row["q2"] != "~":
                quali_time_str = ycd_quali_row["q2"]
            if "q3" in ycd_quali_source.columns and ycd_quali_row["q3"] != "~":
                quali_time_str = ycd_quali_row["q3"]
        else:
            quali_pos_str = ""
            quali_time_str = ""
        if ycd_fastest_lap_data.shape[0] > 0:
            ycd_fastest_lap_data_row = ycd_fastest_lap_data.iloc[0]
            if np.isnan(ycd_fastest_lap_data_row["fastest_lap_time_millis"]):
                fastest_lap_str = ""
            else:
                fastest_lap_str = ycd_fastest_lap_data_row["fastest_lap_time_str"] + " (" + \
                                  int_to_ordinal(ycd_fastest_lap_data_row["rank"]) + " fastest this race)"
            avg_lap_time_str = millis_to_str(ycd_fastest_lap_data_row["avg_lap_time_millis"])
        else:
            fastest_lap_str = ""
            avg_lap_time_str = ""

        driver_name = get_driver_name(driver_id)
        ycc_stats += subheader_template.format(driver_name)
        if wdc_impact_str != "":
            if round_num == 1:
                ycc_stats += template.format("WDC Position: ".ljust(22), wdc_impact_str)
            else:
                ycc_stats += template.format("WDC Impact: ".ljust(22), wdc_impact_str)
        if ycd_pit_stop_data.shape[0] > 0:
            ycc_stats += template.format("Num Pit Stops: ".ljust(22), num_pit_stops_str)
        if quali_pos_str != "":
            ycc_stats += template.format("Qualifying Position: ".ljust(22), quali_pos_str)
        if quali_time_str != "":
            ycc_stats += template.format("Qualifying Time: ".ljust(22), quali_time_str)
        if ycd_results.shape[0] > 0:
            ycc_stats += template.format("Grid Position: ".ljust(22), grid_str)
            ycc_stats += template.format("Finish Position: ".ljust(22), fp_str)
            ycc_stats += template.format("Num Laps: ".ljust(22), laps_str)
            ycc_stats += template.format("Race Time: ".ljust(22), runtime_str)
            ycc_stats += template.format("Points Earned: ".ljust(22), points_str)
        if ycd_fastest_lap_data.shape[0] > 0:
            if fastest_lap_str != "":
                ycc_stats += template.format("Fastest Lap Time: ".ljust(22), fastest_lap_str)
            ycc_stats += template.format("Avg. Lap Time: ".ljust(22), avg_lap_time_str)

    divider = vdivider()
    return row([image_view, divider, Div(text=ycc_stats)], sizing_mode="stretch_both")


def generate_muted_dids(race_results, driver_ids, constructor_id, consider_window):
    """
    Generates a list of driver IDs to have muted within the given consider window.
    :param race_results: Race results
    :param driver_ids: Driver IDs
    :param constructor_id: Constructor ID
    :param consider_window: Consider window
    :return: Driver ID
    """
    constructor_results = race_results[race_results["constructorId"] == constructor_id]
    if constructor_results.shape[0] == 0:
        return set()
    considering_dids = set()
    for driver_id in driver_ids:
        results_row = constructor_results[constructor_results["driverId"] == driver_id]
        if results_row.shape[0] > 0:
            fp = results_row["positionOrder"].values[0]
            if fp > consider_window:
                min_position = fp - consider_window
                max_position = fp + consider_window
            else:
                min_position = 1
                max_position = 2 * consider_window + 1
            considering_slice = race_results[(race_results["positionOrder"] >= min_position) &
                                             (race_results["positionOrder"] <= max_position)]
            considering_dids.update(set(considering_slice["driverId"].unique()))
    all_dids = set(race_results["driverId"])
    muted_dids = all_dids - set(considering_dids)
    return muted_dids


def generate_error_layout(year_id, circuit_id, constructor_id):
    """
    Generates error layout in the event that the given constructor didn't compete at the given circuit in the given
    year
    :param year_id: Year
    :param circuit_id: Circuit ID
    :param constructor_id: Constructor ID
    :return: Error layout
    """
    logging.info("Generating error layout")
    constructor_name = get_constructor_name(constructor_id)
    circuit_name = get_circuit_name(circuit_id)
    constructor_results = results[results["constructorId"] == constructor_id]
    rids_constructor_competed_in = constructor_results["raceId"].values.tolist()
    constructor_races = races.loc[rids_constructor_competed_in]
    cids_constructor_competed_at = constructor_races["circuitId"].unique().tolist()

    # Generate the text
    text = f"Unfortunately, {constructor_name} did not compete in a race at {circuit_name} in {year_id}. The constructor " \
           f"competed at the following tracks in the following years:<br>"
    text += "<ul>"
    for circuit_id in cids_constructor_competed_at:
        years = constructor_races[constructor_races["circuitId"] == circuit_id]["year"].unique().tolist()
        years_str = rounds_to_str(years)
        text += f"<li>{get_circuit_name(circuit_id)} ({years_str})</li>"
    text += "</ul><br>"
    layout = Div(text=text)
    return layout


def is_valid_input(year_id, circuit_id, constructor_id):
    """
    Returns whether the given combo of year, circuit, and constructor ID is valid.
    :param year_id: Year ID
    :param circuit_id: Circuit ID
    :param constructor_id: Constructor ID
    :return: True if valid, False otherwise
    """
    year_races = races[races["year"] == year_id]
    race = year_races[year_races["circuitId"] == circuit_id]
    if race.shape[0] == 0:
        return False
    rid = race.index.values[0]
    race_results = results[results["raceId"] == rid]
    ycc_results = race_results[race_results["constructorId"] == constructor_id]
    return ycc_results.shape[0] > 0
