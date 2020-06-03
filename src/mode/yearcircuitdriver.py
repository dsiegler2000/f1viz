import logging
import math
from collections import defaultdict
from bokeh.layouts import column, row
from bokeh.models import Spacer, Div, Span, Label
from data_loading.data_loader import load_results, load_lap_times, load_pit_stops, load_qualifying, load_circuits, \
    load_fastest_lap_data, load_driver_standings, load_races
from mode import yearcircuit
import numpy as np
from utils import PLOT_BACKGROUND_COLOR, get_driver_name, get_race_name, get_circuit_name, plot_image_url, \
    get_constructor_name, int_to_ordinal, result_to_str, millis_to_str, vdivider, rounds_to_str

# Note, ycd=yearcircuitdriver and dr=driver race

results = load_results()
lap_times = load_lap_times()
pit_stop_data = load_pit_stops()
quali = load_qualifying()
circuits = load_circuits()
fastest_lap_data = load_fastest_lap_data()
driver_standings = load_driver_standings()
races = load_races()


def get_layout(year_id=-1, circuit_id=-1, driver_id=-1, download_image=True, **kwargs):
    # Generate slices
    year_races = races[races["year"] == year_id]
    race = year_races[year_races["circuitId"] == circuit_id]
    if race.shape[0] == 0:
        return generate_error_layout(year_id=year_id, circuit_id=circuit_id, driver_id=driver_id)
    rid = race.index.values[0]
    race_results = results[results["raceId"] == rid]
    ycd_results = race_results[race_results["driverId"] == driver_id]

    logging.info(f"Generating layout for mode YEARCIRCUITDRIVER in yearcircuitdriver, year_id={year_id}, "
                 f"circuit_id={circuit_id}, driver_id={driver_id}")

    if ycd_results.shape[0] == 0:
        return generate_error_layout(year_id=year_id, circuit_id=circuit_id, driver_id=driver_id)

    # Generate more slices
    race_laps = lap_times[lap_times["raceId"] == rid]
    ycd_laps = race_laps[race_laps["driverId"] == driver_id]
    ycd_pit_stop_data = pit_stop_data[(pit_stop_data["raceId"] == rid) & (pit_stop_data["driverId"] == driver_id)]
    race_quali = quali[quali["raceId"] == rid]
    ycd_fastest_lap_data = fastest_lap_data[(fastest_lap_data["raceId"] == rid) &
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
    disclaimer_sc = detect_mark_safety_car(race_laps, race, race_results, plots)

    # Mark fastest lap
    mark_fastest_lap(ycd_results, plots)

    plots = [gap_plot, lap_time_plot]

    # Mark overtakes
    disclaimer_overtakes = detect_mark_overtakes(ycd_laps, race_laps, plots)

    # Mark pit stops
    mark_pit_stops(ycd_pit_stop_data, [gap_plot, lap_time_plot], driver_id)

    # Quali table
    quali_table, quali_source = generate_quali_table(race_quali, race_results, driver_id)

    # Stats layout
    stats_layout = generate_stats_layout(ycd_results, ycd_pit_stop_data, ycd_fastest_lap_data, year_driver_standings,
                                         race_results, quali_source, rid, circuit_id, driver_id,
                                         download_image=download_image)

    driver_name = get_driver_name(driver_id)
    race_name = get_race_name(rid, include_year=True)
    header = Div(text=f"<h2><b>What did {driver_name}'s {race_name} look like?</b></h2><br><i>Yellow dashed "
                      f"vertical lines show the start of a safety car period, orange vertical lines show the end*. "
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

    logging.info("Finished generating layout for mode YEARCIRCUITDRIVER")

    return layout


def generate_gap_plot(race_laps, race_results, driver_id, consider_window=2):
    """
    Plots gap to leader.
    :param race_laps: Race laps
    :param race_results: Race results
    :param driver_id: Driver ID
    :param consider_window: Window to focus on (places around the driver)
    :return: Gap plot layout, cached driver map
    """
    muted_dids = generate_muted_dids(race_results, driver_id, consider_window)
    return yearcircuit.generate_gap_plot(race_laps, race_results, highlight_dids=[driver_id], muted_dids=muted_dids)


def generate_position_plot(race_laps, race_results, cached_driver_map, driver_id, consider_window=2):
    """
    Generates a position plot (position vs lap)
    :param race_laps: Race laps
    :param race_results: Race results
    :param cached_driver_map: Cached driver map
    :param driver_id: Driver ID
    :param consider_window: Consider window
    :return: Position plot
    """
    muted_dids = generate_muted_dids(race_results, driver_id, consider_window)
    return yearcircuit.generate_position_plot(race_laps, cached_driver_map, highlight_dids=[driver_id],
                                              muted_dids=muted_dids)


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


def detect_mark_overtakes(ycd_laps, race_laps, plots):
    """
    Detects and marks overtakes
    :param ycd_laps: YCD laps
    :param race_laps: Race laps
    :param plots: Plots to mark
    :return: Returns a disclaimer div
    """
    if ycd_laps.shape[0] <= 1:
        return Div()

    def mark_overtake(lap, color, names):
        label_kwargs = dict(render_mode="canvas",
                            text_color=color,
                            text_font_size="9pt",
                            angle=0.4 * math.pi)
        line = Span(location=lap, dimension="height", line_color=color, line_width=1.5)
        for p in plots:
            y = p.y_range.start + 0.2 * (p.y_range.end - p.y_range.start)
            dy = 0.1 * (p.y_range.end - p.y_range.start)
            p.renderers.extend([line])
            for name in names:
                label = Label(x=lap + 0.5, y=y, text=name, **label_kwargs)
                p.add_layout(label)
                y += dy

    prev_position = ycd_laps["position"].values[0]
    prev_lap_laps = None
    overtakes = 0
    overtaken = 0
    for idx, row in ycd_laps.iterrows():
        curr_position = row["position"]
        lap = row["lap"]
        lap_laps = race_laps[race_laps["lap"] == row["lap"]]
        if prev_lap_laps is None:
            prev_lap_laps = lap_laps
        if curr_position != prev_position:
            if curr_position < prev_position:  # Overtook people
                names = []
                for opponent_prev_pos in range(prev_position - 1, 0, -1):
                    did = prev_lap_laps[prev_lap_laps["position"] == opponent_prev_pos]
                    if did.shape[0] > 0:
                        did = did["driverId"].values[0]
                        opponent_curr_pos = lap_laps[lap_laps["driverId"] == did]
                        if opponent_curr_pos.shape[0] == 0:
                            opponent_curr_pos = 100
                        else:
                            opponent_curr_pos = opponent_curr_pos["position"].values[0]
                        if opponent_prev_pos < prev_position and opponent_curr_pos > curr_position:
                            overtakes += 1
                            names.append(get_driver_name(did, include_flag=False, just_last=True))
                mark_overtake(lap, "green", names)
            if curr_position > prev_position:  # Got overtaken
                names = []
                for opponent_prev_pos in range(prev_position, 30):
                    did = prev_lap_laps[prev_lap_laps["position"] == opponent_prev_pos]
                    if did.shape[0] > 0:
                        did = did["driverId"].values[0]
                        opponent_curr_pos = lap_laps[lap_laps["driverId"] == did]
                        if opponent_curr_pos.shape[0] == 0:
                            opponent_curr_pos = 100
                        else:
                            opponent_curr_pos = opponent_curr_pos["position"].values[0]
                        if opponent_prev_pos > prev_position and opponent_curr_pos < curr_position:
                            overtaken += 1
                            names.append(get_driver_name(did, include_flag=False, just_last=True))
                    mark_overtake(lap, "red", names)
        prev_position = curr_position
        prev_lap_laps = lap_laps
    disclaimer = Div(text="<b>** Overtakes are detected using lap timing data, and thus may not always be perfectly "
                          "accurate, especially if there are multiple overtakes occurring in one lap. These overtakes "
                          "also include overtakes that occur in the pits.</b>")

    return disclaimer


def mark_pit_stops(ycd_pit_stop_data, plots, driver_id, cached_driver_map=None, h_pct=0.5, show_name=False):
    """
    Marks pit stops with a vertical line
    :param ycd_pit_stop_data: YCD pit stop data
    :param plots: Plots to mark
    :param driver_id: Driver ID
    :param cached_driver_map: Must be passed if `show_name is True`
    :param h_pct: Percent of the height to write the safety car time
    :param show_name: Whether to show the driver name
    :return: Line dict, which maps driver ID to a list of pit stop lines for use on the legend
    """
    if ycd_pit_stop_data.shape[0] == 0:
        return
    line_dict = defaultdict(list)
    for idx, row in ycd_pit_stop_data.iterrows():
        lap = row["lap"]
        millis = row["milliseconds"]
        if np.isnan(millis):
            continue
        time_str = str(round(millis / 1000, 3)) + "s"

        label_kwargs = dict(render_mode="canvas",
                            text_color="hotpink",
                            text_font_size="10pt",
                            text_alpha=0.7,
                            angle=0.4 * math.pi)

        if driver_id and cached_driver_map:
            line_dash = cached_driver_map[driver_id][3]
        else:
            line_dash = "solid"
        line_kwargs = dict(
            x=[lap, lap],
            y=[-1000, 1000],
            line_color="hotpink",
            line_width=2,
            line_dash=line_dash,
            line_alpha=0.7
        )
        driver_name = get_driver_name(driver_id, include_flag=False, just_last=True)
        for p in plots:
            r = (p.y_range.end - p.y_range.start)
            y = p.y_range.start + h_pct * r
            dy = 0.09 * r
            time_label = Label(x=lap + 0.5, y=y, text=time_str, **label_kwargs)
            line = p.line(**line_kwargs)
            line_dict[driver_id].append(line)
            p.add_layout(time_label)
            if show_name:
                name_label = Label(x=lap + 0.6, y=y - dy, text=driver_name, **label_kwargs)
                p.add_layout(name_label)
    return line_dict


def mark_fastest_lap(ycd_results, plots):
    """
    Marks fastest lap with a vertical line
    :param ycd_results: YCD results
    :param plots: Plots to mark
    :return: None
    """
    yearcircuit.mark_fastest_lap(ycd_results, plots)


def generate_lap_time_plot(race_laps, race_results, cached_driver_map, driver_id, consider_window=1):
    """
    Generates a plot of lap time vs laps
    :param race_laps: Race laps
    :param race_results: Race results
    :param cached_driver_map: Cached driver map
    :param driver_id: Driver ID
    :param consider_window: Window to focus on (places around the driver)
    :return: Lap time plot layout
    """
    muted_dids = generate_muted_dids(race_results, driver_id, consider_window)
    kwargs = dict(
        stdev_range=(1.5, 0.2),
        include_hist=False,
        highlight_dids=[driver_id],
        muted_dids=muted_dids
    )
    return yearcircuit.generate_lap_time_plot(race_laps, cached_driver_map, **kwargs)[1]


def generate_quali_table(race_quali, race_results, driver_id):
    """
    Generates qualifying table with driver highlighted.
    :param race_quali: Race qualifying data
    :param race_results: Race results
    :param driver_id: Driver ID
    :return: Quali table, quali source
    """
    quali_table, pole_info, quali_source = yearcircuit.generate_quali_table(race_quali, race_results,
                                                                            highlight_dids=[driver_id])
    return quali_table, quali_source


def generate_stats_layout(ycd_results, ycd_pit_stop_data, ycd_fastest_lap_data, year_driver_standings, race_results,
                          quali_source, race_id, circuit_id, driver_id, download_image=True):
    """
    Stats div including:
    - Location
    - Date
    - Weather
    - Rating
    - Constructor
    - Qualifying position and time
    - Laps
    - Fastest lap time along with rank
    - Average lap time
    - Basic teammate info (i.e. teammate finish in 5th with an average lap time of 1:15.125)
    - Finish position
    - Finish time
    - Points scored
    - Num pit stops
    - WDC impact (WDC place before, WDC after)
    :param ycd_results: YCD results
    :param ycd_pit_stop_data: YCD pit stop data
    :param ycd_fastest_lap_data: YCD fastest lap data
    :param year_driver_standings: YCD driver standings
    :param race_results: Race results
    :param quali_source: Quali source
    :param race_id: Race ID
    :param circuit_id: Circuit ID
    :param driver_id: Driver ID
    :param download_image: Whether to actually download the image
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
        constructor_id = ycd_results_row["constructorId"]
        constructor_str = get_constructor_name(constructor_id)
        grid_str = int_to_ordinal(ycd_results_row["grid"]).strip()
        fp_str, _ = result_to_str(ycd_results_row["positionOrder"], ycd_results_row["statusId"])
        fp_str = fp_str.strip()
        laps_str = str(ycd_results_row["laps"])
        runtime_str = millis_to_str(ycd_results_row["milliseconds"]).strip()
        points = ycd_results_row["points"]
        if abs(int(points) - points) < 0.01:
            points = int(points)
        points_str = str(points)
        teammates = set(race_results[race_results["constructorId"] == constructor_id]["driverId"].values) - {driver_id}
        teammate_strs = []
        for teammate_did in teammates:
            teammate_result = race_results[race_results["driverId"] == teammate_did]
            if teammate_result.shape[0] > 0:
                tm_result_row = teammate_result.iloc[0]
                tm_name = get_driver_name(teammate_did)
                tm_fp_str, _ = result_to_str(tm_result_row["positionOrder"], tm_result_row["statusId"])
                tm_time_str = millis_to_str(tm_result_row["milliseconds"])
                if "ret" in tm_fp_str.lower():
                    teammate_strs.append(tm_name + " " + tm_fp_str)
                else:
                    teammate_strs.append(tm_name + " finished " + tm_fp_str.strip() + " (" + tm_time_str.strip() + ")")
        teammate_str = ", ".join(teammate_strs)
    else:
        constructor_str = ""
        grid_str = ""
        fp_str = ""
        laps_str = ""
        runtime_str = ""
        points_str = ""
        teammate_str = ""
    ycd_quali_source = quali_source[quali_source["driver_id"] == driver_id]
    if ycd_quali_source.shape[0] > 0:
        ycd_quali_row = ycd_quali_source.iloc[0]
        quali_pos = ycd_quali_row["quali_position"]
        quali_pos_str = int_to_ordinal(quali_pos).strip()
        quali_time_str = ""
        if "q1" in ycd_quali_source.columns and ycd_quali_row["q1"] != "":
            quali_time_str = ycd_quali_row["q1"]
        if "q2" in ycd_quali_source.columns and ycd_quali_row["q2"] != "":
            quali_time_str = ycd_quali_row["q2"]
        if "q3" in ycd_quali_source.columns and ycd_quali_row["q3"] != "":
            quali_time_str = ycd_quali_row["q3"]
        quali_time_str = quali_time_str.strip()
    else:
        quali_pos_str = ""
        quali_time_str = ""
    if ycd_fastest_lap_data.shape[0] > 0:
        ycd_fastest_lap_data_row = ycd_fastest_lap_data.iloc[0]
        if np.isnan(ycd_fastest_lap_data_row["fastest_lap_time_millis"]):
            fastest_lap_str = ""
        else:
            fastest_lap_str = ycd_fastest_lap_data_row["fastest_lap_time_str"] + " (" + \
                              int_to_ordinal(ycd_fastest_lap_data_row["rank"]).strip() + " fastest this race)"
        avg_lap_time_str = millis_to_str(ycd_fastest_lap_data_row["avg_lap_time_millis"])
    else:
        fastest_lap_str = ""
        avg_lap_time_str = ""

    header_template = """
    <h2 style="text-align: center;"><b>{}</b></h2>
    """

    template = """
    <pre><b>{}</b> {}<br></pre>
    """

    race_name = get_race_name(race_id, include_year=True)
    driver_name = get_driver_name(driver_id)
    ycd_stats = header_template.format(driver_name + " at the " + race_name)

    ycd_stats += template.format("Circuit Name: ".ljust(22), circuit_str)
    ycd_stats += template.format("Date: ".ljust(22), date_str)
    if weather_str != "" and weather_str.lower() != "nan":
        ycd_stats += template.format("Weather: ".ljust(22), weather_str)
    if not np.isnan(rating):
        ycd_stats += template.format("Rating: ".ljust(22), rating_str)
    if not np.isnan(sc):
        ycd_stats += template.format("Safety Car Laps: ".ljust(22), sc_str)
    if wdc_impact_str != "":
        if round_num == 1:
            ycd_stats += template.format("WDC Position: ".ljust(22), wdc_impact_str)
        else:
            ycd_stats += template.format("WDC Impact: ".ljust(22), wdc_impact_str)
    if ycd_pit_stop_data.shape[0] > 0:
        ycd_stats += template.format("Num Pit Stops: ".ljust(22), num_pit_stops_str)
    if quali_pos_str != "":
        ycd_stats += template.format("Qualifying Position: ".ljust(22), quali_pos_str)
    if quali_time_str != "":
        ycd_stats += template.format("Qualifying Time: ".ljust(22), quali_time_str)
    if ycd_results.shape[0] > 0:
        ycd_stats += template.format("Constructor: ".ljust(22), constructor_str)
        ycd_stats += template.format("Grid Position: ".ljust(22), grid_str)
        ycd_stats += template.format("Finish Position: ".ljust(22), fp_str)
        ycd_stats += template.format("Num Laps: ".ljust(22), laps_str)
        ycd_stats += template.format("Race Time: ".ljust(22), runtime_str)
        ycd_stats += template.format("Points Earned: ".ljust(22), points_str)
        ycd_stats += template.format("Teammate(s): ".ljust(22), teammate_str)
    if ycd_fastest_lap_data.shape[0] > 0:
        if fastest_lap_str != "":
            ycd_stats += template.format("Fastest Lap Time: ".ljust(22), fastest_lap_str)
        ycd_stats += template.format("Avg. Lap Time: ".ljust(22), avg_lap_time_str)

    divider = vdivider()
    return row([image_view, divider, Div(text=ycd_stats)], sizing_mode="stretch_both")


def generate_muted_dids(race_results, driver_id, consider_window):
    """
    Generates the set of muted driver IDs.
    :param race_results: Race results
    :param driver_id: Driver ID
    :param consider_window: Consider window
    :return: Muted driver ID set
    """
    results_row = race_results[race_results["driverId"] == driver_id]
    if results_row.shape[0] == 0:
        return Div()
    else:
        fp = results_row["positionOrder"].values[0]
        cid = results_row["constructorId"].values[0]
    if fp > consider_window:
        min_position = fp - consider_window
        max_position = fp + consider_window
    else:
        min_position = 1
        max_position = 2 * consider_window + 1
    considering_dids = race_results[(race_results["positionOrder"] >= min_position) &
                                    (race_results["positionOrder"] <= max_position)]["driverId"].unique().tolist()
    all_dids = set(race_results["driverId"])
    teammates = set(race_results[race_results["constructorId"] == cid]["driverId"].values) - {driver_id}
    muted_dids = all_dids - set(considering_dids) - teammates
    return muted_dids


def generate_error_layout(year_id, circuit_id, driver_id):
    """
    Generates an error layout in the event that there was no race in the given year at the given circuit or the given
    driver didn't compete in the given race.
    :param year_id: Year
    :param circuit_id: Circuit ID
    :param driver_id: Driver ID
    :return: Div layout
    """
    logging.info("Generating error layout")
    driver_name = get_driver_name(driver_id)
    circuit_name = get_circuit_name(circuit_id)
    driver_results = results[results["driverId"] == driver_id]
    rids_driver_competed_in = driver_results["raceId"].values.tolist()
    driver_races = races.loc[rids_driver_competed_in]
    cids_driver_competed_at = driver_races["circuitId"].unique().tolist()

    # Generate the text
    text = f"Unfortunately, {driver_name} did not compete in a race at {circuit_name} in {year_id}. The driver " \
           f"competed at the following tracks in the following years:<br>"
    text += "<ul>"
    for circuit_id in cids_driver_competed_at:
        years = driver_races[driver_races["circuitId"] == circuit_id]["year"].unique().tolist()
        years_str = rounds_to_str(years)
        text += f"<li>{get_circuit_name(circuit_id)} ({years_str})</li>"
    text += "</ul><br>"
    layout = Div(text=text)
    return layout


def is_valid_input(year_id, circuit_id, driver_id):
    """
    Returns whether the given combo of year, circuit, and driver ID is valid.
    :param year_id: Year ID
    :param circuit_id: Circuit ID
    :param driver_id: Driver ID
    :return: True if valid, False otherwise
    """
    race = races[(races["circuitId"] == circuit_id) & (races["year"] == year_id)]
    if race.shape[0] == 0:
        return False
    rid = race.index.values[0]
    race_results = results[results["raceId"] == rid]
    ycd_results = race_results[race_results["driverId"] == driver_id]
    return ycd_results.shape[0] > 0
