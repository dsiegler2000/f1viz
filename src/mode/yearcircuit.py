import logging
from math import isnan
from bokeh.layouts import column, row
from bokeh.plotting import figure, Figure
from data_loading.data_loader import load_circuits, load_lap_times, load_races, load_drivers, load_results, \
    load_constructors, load_pit_stops, load_qualifying, load_status, load_driver_standings, load_constructor_standings, \
    load_fastest_lap_data
import pandas as pd
import numpy as np
from bokeh.models import HoverTool, Div, Legend, LegendItem, ColumnDataSource, Range1d, Spacer, \
    CrosshairTool, DataRange1d, Span, TableColumn, DataTable, DatetimeTickFormatter, HTMLTemplateFormatter
from bokeh.models.tickers import FixedTicker

from mode import driver
from utils import get_line_thickness, ColorDashGenerator, plot_image_url, DATETIME_TICK_KWARGS, int_to_ordinal, \
    get_race_name, result_to_str, position_text_to_str
from utils import millis_to_str, get_driver_name, get_constructor_name, PLOT_BACKGROUND_COLOR, \
    vdivider, hdivider
from datetime import datetime

circuits = load_circuits()
lap_times = load_lap_times()
races = load_races()
drivers = load_drivers()
results = load_results()
constructors = load_constructors()
pit_stops = load_pit_stops()
qualifying = load_qualifying()
statuses = load_status()
driver_standings = load_driver_standings()
constructor_standings = load_constructor_standings()
fastest_lap_data = load_fastest_lap_data()

UP_ARROW = "^"
DOWN_ARROW = "v"
SIDE_ARROW = ">"

# TODO
#  remove axis sharing!                                                                                         √
#  fix lap time plot y axis/panning                                                                             √
#  add ordinals                                                                                                 √
#  why is the gap plot off by a bit sometimes? (see brazil 2019, the very end)                                  √
#  figure out why all these warnings happen (try playing with gap plot)?                                        √
#  add SP v FP and MLTR vs FP

# TODO
#   Make sure tables are sortable                                                                           √
#   Make sure second axes are scaled properly                                                               √
#   Make sure using ordinals (1st, 2nd, 3rd) on everything                                                  √
#   Make sure the mode has a header                                                                         √


def get_layout(year_id=-1, circuit_id=-1, **kwargs):
    # Get some useful slices
    race = races[(races["circuitId"] == circuit_id) & (races["year"] == year_id)]
    if race.shape[0] == 0:
        return generate_error_layout(year_id, circuit_id)

    race_id = race.index.values[0]
    race_laps = lap_times[lap_times["raceId"] == race_id]
    race = races[races.index == race_id]
    race_results = results[results["raceId"] == race_id]
    race_pit_stops = pit_stops[pit_stops["raceId"] == race_id]
    race_quali = qualifying[qualifying["raceId"] == race_id]
    race_driver_standings = driver_standings[driver_standings["raceId"] == race_id]
    race_constructor_standings = constructor_standings[constructor_standings["raceId"] == race_id]
    race_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"] == race_id]

    if race_results.shape[0] == 0:
        return Div(text="Unfortunately, we have no data on this race.")

    logging.info(f"Generating layout for mode YEARCIRCUIT in yearcircuit, year_id={year_id}, "
                 f"circuit_id={circuit_id}, race_id={race_id}")

    # Generate the gap plot
    gap_plot, cached_driver_map = generate_gap_plot(race_laps, race_results)

    # Generate position plot
    position_plot = generate_position_plot(race_laps, cached_driver_map)

    # Generate the lap time plot
    lap_time_plot_layout, lap_time_plot = generate_lap_time_plot(race_laps, cached_driver_map)

    # Generate the pit stop plot
    pit_stop_plot = generate_pit_stop_plot(race_pit_stops, cached_driver_map, race_laps)

    all_plots = [gap_plot, position_plot, lap_time_plot, pit_stop_plot]

    # Start position vs finish position
    spvfp_scatter = generate_spvfp_scatter(race_results, race, race_driver_standings)

    # Mean lap time rank vs finish position
    mltr_fp_scatter = generate_mltr_fp_scatter(race_results, race, race_driver_standings)

    # Mark safety car and fastest lap
    sc_disclaimer_div = detect_mark_safety_car_end(race_laps, race, race_results, all_plots)

    # Generate race stats
    race_stats_layout = generate_stats_layout(race_quali, race_results, race_laps, circuit_id,
                                              race_driver_standings, race_constructor_standings,
                                              race_fastest_lap_data, race_id)

    # Create a header
    title = get_race_name(race_id, include_year=True)
    header = Div(text=f"<h2><b>What did the {title} look like?</b></h2><br><i>Yellow dashed vertical lines show the "
                      f"start of a safety car period, orange vertical lines show the end*.</i>")

    # Bring it all together
    middle_spacer = Spacer(width=5, background=PLOT_BACKGROUND_COLOR)
    layout = column(header,
                    gap_plot, middle_spacer,
                    position_plot, middle_spacer,
                    lap_time_plot_layout, middle_spacer,
                    pit_stop_plot, middle_spacer,
                    row([spvfp_scatter, mltr_fp_scatter], sizing_mode="stretch_width"),
                    sc_disclaimer_div,
                    race_stats_layout,
                    sizing_mode="stretch_width")

    logging.info("Finished generating layout for mode YEARCIRCUIT")

    return layout


def generate_stats_layout(race_quali, race_results, race_laps, circuit_id, race_driver_standings,
                          race_constructor_standings, race_fastest_lap_data, race_id):
    """
    Generates a Wikipedia-style Div with lots of facts on the race, including tables on qualifying and fastest lap.
    :param race_quali: Race quali
    :param race_results: Race results
    :param race_laps: Race laps
    :param circuit_id: Circuit ID
    :param race_driver_standings: Race driver standings
    :param race_constructor_standings: Race constructor standings
    :param race_fastest_lap_data: Race fastest lap data
    :param race_id: Race ID
    :return: Race stats layout
    """
    logging.info("Generating race stats layout")

    # Track image
    image_url = str(circuits.loc[circuit_id, "imgUrl"])
    image_view = plot_image_url(image_url)
    disclaimer = Div(text="The image is of the current configuration of the track.")
    image_view = column([image_view, disclaimer], sizing_mode="stretch_both")

    # Qualifying
    quali_table, pole_info, _ = generate_quali_table(race_quali, race_results)

    # Final standings, including reason for DNF and grid position
    results_table, podium_info = generate_results_table(race_results)

    # Championship standings after race - WDC
    wdc_impact_table = generate_wdc_impact_table(race_driver_standings, race_id)

    # Championship standings after race - constructors
    constructors_impact_table = generate_wcc_impact_table(race_constructor_standings, race_id)

    championship_impact_title = Div(text="<h2><b>Championship Standings after Race</b></h2>")

    divider = hdivider(top_border_thickness=16, bottom_border_thickness=2, line_thickness=1)
    impacts = column([championship_impact_title, wdc_impact_table, divider, constructors_impact_table],
                     sizing_mode="stretch_both")

    # Fastest lap for every driver and average lap time
    fastest_lap_table, fastest_lap_info = generate_fastest_lap_table(race_results, race_laps, race_fastest_lap_data)

    # Random stats: weather, SC, fan rating, link to Wikipedia and F1 TV
    race = races.loc[race_id]
    circuit = circuits.loc[circuit_id]
    date = race["datetime"]
    location = circuit["location"] + ", " + circuit["country"]
    runtime = millis_to_str(race["runtime"], fallback=None)
    weather = race["weather"]
    sc = race["SCLaps"]
    rating = race["rating"]

    header_template = """
    <h2 style="text-align: center;"><b>{}</b></h2>
    """

    template = """
    <pre><b>{}</b> {}<br></pre>
    """

    race_stats = header_template.format("Race Details")
    if isinstance(date, str):
        date = date.split(" ")[0]
        if "1990-01-01" not in date:
            date = datetime.strptime(date, "%Y-%m-%d").strftime("%d %B %Y")
            race_stats += template.format("Date: ".ljust(13), date)
    if isinstance(location, str):
        race_stats += template.format("Location: ".ljust(13), location)
    if isinstance(runtime, str):
        race_stats += template.format("Runtime: ".ljust(13), runtime)
    if isinstance(weather, str):
        race_stats += template.format("Weather: ".ljust(13), weather)
    if not np.isnan(sc):
        race_stats += template.format("SC Laps: ".ljust(13), int(sc))
    if not np.isnan(rating):
        race_stats += template.format("Fan Rating: ".ljust(13), str(rating) + " / 10")
    if len(pole_info) > 0:
        race_stats += header_template.format("Pole Position")
        pole_position_driver = pole_info[0]
        pole_position_constructor = get_constructor_name(pole_info[1])
        pole_position_time = pole_info[2]
        race_stats += template.format("Driver: ".ljust(13), pole_position_driver)
        race_stats += template.format("Constructor: ".ljust(13), pole_position_constructor)
        race_stats += template.format("Time: ".ljust(13), pole_position_time)
    if len(fastest_lap_info) > 0:
        race_stats += header_template.format("Fastest Lap")
        fastest_lap_driver = fastest_lap_info[0]
        fastest_lap_constructor = fastest_lap_info[1]
        fastest_lap_time = fastest_lap_info[2]
        fastest_lap_number = fastest_lap_info[3]
        race_stats += template.format("Driver: ".ljust(13), fastest_lap_driver)
        race_stats += template.format("Constructor: ".ljust(13), fastest_lap_constructor)
        race_stats += template.format("Time: ".ljust(13), fastest_lap_time + " on lap " + str(fastest_lap_number))
    if len(podium_info) > 0:
        race_stats += header_template.format("Podium")
        race_stats += template.format("First: ".ljust(13), podium_info[0][0] + " (" + podium_info[0][1]) + ")"
        race_stats += template.format("Second: ".ljust(13), podium_info[1][0] + " (" + podium_info[1][1]) + ")"
        race_stats += template.format("Third: ".ljust(13), podium_info[2][0] + " (" + podium_info[2][1]) + ")"

    stats_div = Div(text=race_stats)

    divider = vdivider()
    return column([row([image_view, divider, quali_table], sizing_mode="stretch_both"),
                   row([impacts, divider, results_table], sizing_mode="stretch_both"),
                   row([fastest_lap_table, divider, stats_div], sizing_mode="stretch_both"),
                   Div(height=100)], sizing_mode="stretch_width")


def generate_fastest_lap_table(race_results, race_laps, race_fastest_lap_data):
    """
    Generates a table that shows drivers' fastest laps and average lap times.
    :param race_results: Race results
    :param race_laps: Race laps
    :param race_fastest_lap_data: Race fastest lap data
    :return: Fastest lap table layout, a tuple containing some info on the fastest lap
    """
    if race_laps["milliseconds"].isna().sum() == race_laps.shape[0] or race_laps.shape[0] == 0:
        return Div(text="Unfortunately, we do not have any lap timing data on this race and thus we don't have the "
                        "fastest lap information (yet). We have lap timing data for the 1996 season onwards."), ()

    title_div = Div(text="<h2><b>Fastest and Average Lap Times</b></h2>")
    fastest_lap_columns = [
        TableColumn(field="rank", title="Rank", width=75),
        TableColumn(field="name", title="Driver", width=200),
        TableColumn(field="constructor_name", title="Constructor", width=200),
        TableColumn(field="fastest_lap_time_str", title="Fastest Lap Time", width=150),
        TableColumn(field="avg_lap_time_str", title="Mean Lap Time", width=150),
    ]

    race_fastest_lap_data = race_fastest_lap_data.copy()
    race_fastest_lap_data["rank"] = race_fastest_lap_data["rank"].replace("  ", "~").apply(int_to_ordinal)
    race_fastest_lap_data = race_fastest_lap_data.sort_values(by="rank")
    race_fastest_lap_data["fastest_lap_time_str"] = race_fastest_lap_data["fastest_lap_time_str"].fillna("")
    race_fastest_lap_data["avg_lap_time_str"] = race_fastest_lap_data["avg_lap_time_str"].fillna("")
    race_fastest_lap_data["rank"] = race_fastest_lap_data["rank"].replace("UNK", "  ")

    fast_source = ColumnDataSource(data=race_fastest_lap_data)
    fast_table = DataTable(source=fast_source, columns=fastest_lap_columns, index_position=None,
                           min_height=min(530, 30 * race_fastest_lap_data.shape[0]))
    fast_row = row([fast_table], sizing_mode="stretch_width")

    c = [title_div, fast_row]

    use_lap_data = race_results["rank"].isna().sum() == race_results.shape[0] or race_results["rank"].sum() < 0.1
    use_lap_data = use_lap_data or race_results["rank"].unique().shape[0] == 1

    if use_lap_data:
        disclaimer = Div(text="This data was generated based off of lap timing data. This means that the "
                              "fastest lap ranking may occasionally be inaccurate.")
        c.append(disclaimer)

    fastest_lap = race_fastest_lap_data[race_fastest_lap_data["rank"].str.match(" 1")]
    fastest_did = fastest_lap["driver_id"].values[0]
    fastest_lap_lap = race_results[race_results["driverId"] == fastest_did]["fastestLap"].values[0]
    fastest_lap_lap = None if np.isnan(fastest_lap_lap) else int(fastest_lap_lap)
    fastest_lap_info = (fastest_lap["name"].values[0], fastest_lap["constructor_name"].values[0],
                        fastest_lap["fastest_lap_time_str"].values[0], fastest_lap_lap)

    return column(c, sizing_mode="stretch_width"), fastest_lap_info


def generate_wcc_impact_table(race_constructor_standings, race_id):
    """
    Generates a table that describes the impact this race had on the constructors's championship.
    :param race_constructor_standings: Constructor's standings for this race
    :param race_id: Race ID of this race
    :return: The table layout
    """
    title_div = Div(text="<h3><b>Constructors's Championship Standings</b></h3>")

    data = {
        "position": [],
        "name": [],
        "points": []
    }

    # Let's see if there was a race before this in the championship
    race_round = races.loc[race_id, "round"]
    race_before = race_round > 1
    if race_before:
        year = races.loc[race_id, "year"]
        rid_before = races[(races["year"] == year) & (races["round"] == race_round - 1)].index.values[0]
        before_constructor_standings = constructor_standings[constructor_standings["raceId"] == rid_before]
        data["position_change"] = []
    else:
        before_constructor_standings = None

    for idx, standings_row in race_constructor_standings.sort_values(by="position").iterrows():
        constructor_id = standings_row["constructorId"]
        position = standings_row["position"]
        position = "UNRANKED" if np.isnan(position) else str(int(position)).rjust(2)
        name = get_constructor_name(constructor_id)
        points = standings_row["points"]
        points = 0 if np.isnan(points) else int(points)

        if race_before:
            pred = before_constructor_standings["constructorId"] == constructor_id
            position_before = before_constructor_standings[pred]["position"]
            if position_before.shape[0] == 0:
                position_change = SIDE_ARROW + " "
            else:
                position_before = position_before.values[0]
                position_now = standings_row["position"]
                position_diff = int(position_before - position_now)
                if position_diff == 0:
                    position_change = SIDE_ARROW + " "
                elif position_diff > 0:
                    position_change = UP_ARROW + " " + str(position_diff)
                else:
                    position_change = DOWN_ARROW + " " + str(abs(position_diff))
            data["position_change"].append(position_change)

        data["position"].append(int_to_ordinal(position).rjust(4))
        data["name"].append(name)
        data["points"].append(str(points).rjust(2))

    impact_columns = [
        TableColumn(field="position", title="Pos.", width=75),
        TableColumn(field="name", title="Constructor", width=200),
        TableColumn(field="points", title="Points", width=75),
    ]

    if race_before:
        impact_columns.insert(0, TableColumn(field="position_change", title="Change", width=75))

    impact_source = ColumnDataSource(data=data)
    impact_table = DataTable(source=impact_source, columns=impact_columns, index_position=None, height=170)
    impact_row = row([impact_table], sizing_mode="stretch_both")

    return column([title_div, impact_row])


def generate_wdc_impact_table(race_driver_standings, race_id):
    """
    Generates a table that describes the impact this race had on the driver's championship.
    :param race_driver_standings: Driver's standings for this race
    :param race_id: Race ID of this race
    :return: The table layout
    """
    title_div = Div(text="<h3><b>Driver's Championship Standings</b></h3>")

    data = {
        "position": [],
        "name": [],
        "points": []
    }

    # Let's see if there was a race before this in the championship
    race_round = races.loc[race_id, "round"]
    race_before = race_round > 1
    if race_before:
        year = races.loc[race_id, "year"]
        rid_before = races[(races["year"] == year) & (races["round"] == race_round - 1)].index.values[0]
        before_driver_standings = driver_standings[driver_standings["raceId"] == rid_before]
        data["position_change"] = []
    else:
        before_driver_standings = None

    for idx, standings_row in race_driver_standings.sort_values(by="position").iterrows():
        driver_id = standings_row["driverId"]
        position = standings_row["position"]
        position = "UNRANKED" if np.isnan(position) else str(int(position)).rjust(2)
        name = get_driver_name(driver_id)
        points = standings_row["points"]
        points = 0 if np.isnan(points) else int(points)

        if race_before:
            position_before = before_driver_standings[before_driver_standings["driverId"] == driver_id]["position"]
            if position_before.shape[0] == 0:
                position_change = SIDE_ARROW + " "
            else:
                position_before = position_before.values[0]
                position_now = standings_row["position"]
                position_diff = int(position_before - position_now)
                if position_diff == 0:
                    position_change = SIDE_ARROW + " "
                elif position_diff > 0:
                    position_change = UP_ARROW + " " + str(position_diff)
                else:
                    position_change = DOWN_ARROW + " " + str(abs(position_diff))
            data["position_change"].append(position_change)

        data["position"].append(int_to_ordinal(position).rjust(4))
        data["name"].append(name)
        data["points"].append(str(points).rjust(2))

    impact_columns = [
        TableColumn(field="position", title="Pos.", width=75),
        TableColumn(field="name", title="Driver", width=200),
        TableColumn(field="points", title="Points", width=75),
    ]

    if race_before:
        impact_columns.insert(0, TableColumn(field="position_change", title="", width=75))

    impact_source = ColumnDataSource(data=data)
    impact_table = DataTable(source=impact_source, columns=impact_columns, index_position=None, height=170)
    impact_row = row([impact_table], sizing_mode="stretch_both")

    return column([title_div, impact_row])


def generate_results_table(race_results):
    """
    Generates a table describing the results of this race.
    :param race_results: Race results
    :return: Table layout
    """
    # Finishing position, driver, constructor, laps, time/retired, grid, points
    title_div = Div(text="<h2><b>Race Results</b></h2>")

    data = {
        "name": [],
        "constructor_name": [],
        "grid": [],
        "finishing_position": [],
        "laps": [],
        "timeretired": [],
        "points": []
    }
    include_driver_num = races.loc[race_results["raceId"].values[0], "year"] >= 2014
    if include_driver_num:
        data["driver_num"] = []

    first_time = None
    for idx, result_row in race_results.sort_values(by="positionOrder").iterrows():
        # handle the DNQ cases
        driver_id = result_row["driverId"]
        name = get_driver_name(driver_id)
        constructor_id = race_results[race_results["driverId"] == driver_id]["constructorId"].values[0]
        constructor_name = get_constructor_name(constructor_id)
        grid = result_row["grid"]
        if grid <= 0:
            grid = "PL"
        finishing_position = position_text_to_str(result_row["positionText"])
        finishing_position = finishing_position.rjust(3)
        laps = result_row["laps"]
        laps = str(laps) if laps > 0 else ""
        timeretired = result_row["milliseconds"]
        status = result_row["statusId"]
        status_str = statuses.loc[result_row["statusId"], "status"]
        if status == 1 and "laps" not in status_str.lower():  # Finished within a lap
            if first_time is None:
                first_time = timeretired
                timeretired = millis_to_str(timeretired)
            else:
                delta = timeretired - first_time
                timeretired = "+ " + millis_to_str(delta, format_seconds=True)
                timeretired = timeretired.rjust(10)
        else:  # Either retired or finished more than a lap behind
            timeretired = statuses.loc[result_row["statusId"], "status"]
        points = result_row["points"]
        if float(points) - int(points) > 0.01:
            points = str(float(points)).rjust(2)
        elif points <= 0.01:
            points = ""
        else:
            points = str(int(points))
        driver_num = drivers.loc[driver_id, "number"]

        data["name"].append(name)
        data["constructor_name"].append(constructor_name)
        data["grid"].append(int_to_ordinal(grid).rjust(4))
        data["finishing_position"].append(int_to_ordinal(finishing_position).rjust(4))
        data["laps"].append(str(laps).rjust(2))
        data["timeretired"].append(timeretired)
        data["points"].append(points)
        if include_driver_num:
            data["driver_num"].append(str(driver_num).rjust(2))

    results_columns = [
        TableColumn(field="finishing_position", title="Pos.", width=100),
        TableColumn(field="name", title="Driver", width=200),
        TableColumn(field="constructor_name", title="Constructor", width=200),
        TableColumn(field="laps", title="Laps", width=100),
        TableColumn(field="timeretired", title="Time/Retired", width=150),
        TableColumn(field="grid", title="Grid", width=75),
        TableColumn(field="points", title="Points", width=75),
    ]

    if include_driver_num:
        results_columns.insert(1, TableColumn(field="driver_num", title="No.", width=75))

    data = pd.DataFrame.from_dict(data)
    results_source = ColumnDataSource(data=data)
    results_table = DataTable(source=results_source, columns=results_columns, min_height=530, index_position=None)
    results_row = row([results_table], sizing_mode="stretch_both")

    # Get the podium info
    first = data[data["finishing_position"] == " 1st"]
    second = data[data["finishing_position"] == " 2nd"]
    third = data[data["finishing_position"] == " 3rd"]

    first_name = first["name"].values[0] if first.shape[0] > 0 else None
    second_name = second["name"].values[0] if second.shape[0] > 0 else None
    third_name = third["name"].values[0] if third.shape[0] > 0 else None
    first_constructor_name = first["constructor_name"].values[0] if first.shape[0] > 0 else None
    second_constructor_name = second["constructor_name"].values[0] if second.shape[0] > 0 else None
    third_constructor_name = third["constructor_name"].values[0] if third.shape[0] > 0 else None

    first = (first_name, first_constructor_name)
    second = (second_name, second_constructor_name)
    third = (third_name, third_constructor_name)

    first = "Not awarded" if first == (None, None) else first
    second = "Not awarded" if second == (None, None) else second
    third = "Not awarded" if third == (None, None) else third

    podium_info = (
        first,
        second,
        third
    )

    return column([title_div, results_row]), podium_info


def generate_quali_table(race_quali, race_results, highlight_dids=None):
    """
    Generates a table describing the results of qualifying.
    :param race_quali: Race qualifying
    :param race_results: Race results
    :param highlight_dids: List of driver IDs to highlight
    :return: Table layout, a tuple containing information on the pole driver, full source
    """
    # TODO refactor to use pandas DataFrame
    logging.info("Generating quali table")
    if highlight_dids is None:
        highlight_dids = []
    n = race_quali.shape[0]

    title_div = Div(text="<h2><b>Qualifying Results</b></h2>")

    include_driver_num = races.loc[race_results["raceId"].values[0], "year"] >= 2014

    # Note, we can have nothing, q1 only, q1 and q2 only, and q1, q2, and q3
    q1_happened = int(race_quali["q1"].isna().sum()) != n
    q2_happened = int(race_quali["q2"].isna().sum()) != n
    q3_happened = int(race_quali["q3"].isna().sum()) != n

    have_quali_data = n > 0 and q1_happened

    data = {
        "driver_id": [],
        "name": [],
        "constructor_name": [],
        "final_grid": [],
        "quali_position": []
    }
    if q1_happened:
        data["q1"] = []
    if q2_happened:
        data["q2"] = []
    if q3_happened:
        data["q3"] = []
    if include_driver_num:
        data["driver_num"] = []
    pole = None
    quali_position = 1  # Yes, this is a bit sketchy but consistent
    driver_ids = race_quali["driverId"].unique() if have_quali_data else race_results["driverId"].unique()
    highlight_names = []
    selected_idxs = []
    for driver_id in driver_ids:
        name = get_driver_name(driver_id)
        constructor_id = race_results[race_results["driverId"] == driver_id]["constructorId"].values[0]
        constructor_name = get_constructor_name(constructor_id)
        final_grid = race_results[race_results["driverId"] == driver_id]["grid"].values[0]
        if final_grid <= 0:
            final_grid = "PL"
        driver_num = drivers.loc[driver_id, "number"]

        quali_row = race_quali[race_quali["driverId"] == driver_id]

        time = ""
        if q1_happened:
            # Position, name, constructor, q1, q2, q3 times, final grid position
            time = millis_to_str(quali_row["q1"].values[0])
            time = "~" if time is "" else time
            data["q1"].append(time)
            if q2_happened:
                time = millis_to_str(quali_row["q2"].values[0])
                time = "~" if time is "" else time
                data["q2"].append(time)
                if q3_happened:
                    time = millis_to_str(quali_row["q3"].values[0])
                    time = "~" if time is "" else time
                    data["q3"].append(time)

        data["driver_id"].append(driver_id)
        data["name"].append(name)
        data["quali_position"].append(int_to_ordinal(quali_position).rjust(4))
        data["constructor_name"].append(constructor_name)
        data["final_grid"].append(int_to_ordinal(final_grid).rjust(4))
        if include_driver_num:
            data["driver_num"].append(str(driver_num).rjust(2))

        if quali_position == 1:
            pole = (name, constructor_id, time)

        if driver_id in highlight_dids:
            highlight_names.append(name)
            selected_idxs.append(quali_position - 1)

        quali_position += 1

    template = """
    <div style="font-weight:<%= 
        (function fontweightfromname(){
            if(""" + str(highlight_names) + """.includes(value)){
                return("900")
            } else{ return("bold") }
            }()) %>; 
        font-style:<%= 
        (function fontstylefromname(){
            if(""" + str(highlight_names) + """.includes(value)){
                return("italic")
            } else{ return("normal") }
            }()) %>;
        "> 
    <%= value %></div>
    """
    formatter = HTMLTemplateFormatter(template=template)
    quali_columns = [
        TableColumn(field="name", title="Driver", width=200, formatter=formatter),
        TableColumn(field="constructor_name", title="Constructor", width=175),
        TableColumn(field="final_grid", title="Final Grid", width=100)
    ]

    if have_quali_data:
        quali_columns.insert(0, TableColumn(field="quali_position", title="Pos.", width=50))
        w = int(300 / (q1_happened + q2_happened + q3_happened))
        if q1_happened:
            title = "Q1" if q2_happened else "Time"
            quali_columns.insert(-1, TableColumn(field="q1", title=title, width=w))
        if q2_happened:
            quali_columns.insert(-1, TableColumn(field="q2", title="Q2", width=w))
        if q3_happened:
            quali_columns.insert(-1, TableColumn(field="q3", title="Q3", width=w))
    if include_driver_num:
        quali_columns.insert(1, TableColumn(field="driver_num", title="No.", width=75))

    quali_source = ColumnDataSource(data=data)
    quali_source.selected.indices = selected_idxs
    quali_table = DataTable(source=quali_source, columns=quali_columns, index_position=None)

    # Generate a quick disclaimer about qualifying formats
    disclaimer = Div(text="The qualifying format has changed over the years. The numbers reported are simply "
                          "the raw times and don't take into account any aggregation.")

    quali_row = row([quali_table], sizing_mode="stretch_both")
    return column([title_div, quali_row, disclaimer], sizing_mode="stretch_both"), pole, \
        pd.DataFrame(data)


def mark_fastest_lap(race_results, plots, color="white"):
    """
    Marks the fastest lap of the race on plots with a vertical line in the given color.
    :param race_results: Race results
    :param plots: Plots to mark
    :param color: Color to draw the line in
    :return: None
    """
    # rank, fastest lap
    if race_results["fastestLap"].isna().sum() == race_results.shape[0] or \
            race_results["fastestLapTime"].isna().sum() == race_results.shape[0] or race_results.shape[0] == 0:
        return
    fastest_lap_idx = race_results[race_results["rank"] > 0.9]["rank"].idxmin()
    fastest_lap = int(race_results.loc[fastest_lap_idx, "fastestLap"])
    for p in plots:
        kwargs = dict(
            line_color=color,
            location=fastest_lap,
            dimension="height",
            line_alpha=0.7,
            line_width=3,
        )
        line = Span(**kwargs)
        p.renderers.extend([line])


def detect_mark_safety_car_end(race_laps, race, race_results, plots, draw_finish_line=False):
    """
    Detects safety car periods by comparing the per-lap mean lap time with the whole race mean. Adds lines to mark the
    start and end of safety car periods.
    :param race_laps: Race laps
    :param race: Race
    :param race_results: Race results
    :param plots: Plots to add lines to
    :param draw_finish_line: Whether to draw the finish line
    :return: A "disclaimer" Div.
    """
    # Detect safety cars using mean variance method
    if race_laps.shape[0] == 0:
        return Spacer(width=0, height=0, background=PLOT_BACKGROUND_COLOR)
    race_mean_time = np.median(race_laps["milliseconds"])
    race_std = np.std(race_laps["milliseconds"])
    cutoff = race_mean_time + 2 * race_std

    safety_car_start = []
    safety_car_end = []
    in_safety_car = False
    for lap_num in range(1, race_laps["lap"].max() + 1):
        mean_lap_time = np.mean(race_laps[race_laps["lap"] == lap_num]["milliseconds"])
        if mean_lap_time > cutoff and not in_safety_car and lap_num > 4:
            safety_car_start.append(lap_num - 0.5)
            in_safety_car = True
        if mean_lap_time <= cutoff and in_safety_car:
            safety_car_end.append(lap_num + 0.5)
            in_safety_car = False
    # Quick pass to merge back-to-back detected safety cars
    safety_car_start_merged = []
    safety_car_end_merged = []
    skip_next = False
    for i in range(0, len(safety_car_end)):
        if skip_next:
            skip_next = False
            continue
        if i < len(safety_car_start) - 1 and safety_car_end[i] == safety_car_start[i + 1]:
            safety_car_start_merged.append(safety_car_start[i])
            safety_car_end_merged.append(safety_car_end[i + 1])
            skip_next = True
        else:
            safety_car_start_merged.append(safety_car_start[i])
            safety_car_end_merged.append(safety_car_end[i])
    safety_car_start = safety_car_start_merged
    safety_car_end = safety_car_end_merged

    if in_safety_car:
        safety_car_end.append(race_laps["lap"].max())

    for start, end in zip(safety_car_start, safety_car_end):
        start_line = Span(location=start, dimension="height", line_color="yellow", line_width=3, line_dash="dashed")
        end_line = Span(location=end, dimension="height", line_color="orange", line_width=3, line_dash="dashed")
        for p in plots:
            if isinstance(p, Figure):
                p.renderers.extend([start_line, end_line])
    real_sc_laps = race["SCLaps"].values[0]
    has_sc_laps = real_sc_laps is not "" and not isnan(real_sc_laps)

    text = f"* The safety car laps are detected by comparing the mean lap time of that lap with the " \
           f"overall mean lap time, and thus may not be accurate. Safety cars before lap 4 are not detected and " \
           f"virtual safety cars are inconsistently detected, among other issues. " \
           f"This algorithm detected {len(safety_car_start)} safety car period(s). "
    if has_sc_laps:
        text += f"<br>This race really was under safety car for {int(real_sc_laps)} laps."
    else:
        text += "Unfortunately, we do not have any official data on the safety car laps during the Grand Prix."

    disclaimer = Div(text=f"<b>{text}</b>")

    if draw_finish_line:
        # Find the end of the race
        winner_did = race_results[race_results["position"] == 1]["driverId"].values[0]
        last_lap = race_laps[race_laps["driverId"] == winner_did]["lap"].max()
        for p in plots:
            kwargs = dict(
                location=last_lap - 0.5,
                dimension="height",
                line_alpha=0.5,
                line_width=3,
            )
            start_line1 = Span(line_color="white", line_dash=[30], **kwargs)
            start_line2 = Span(line_color="black", line_dash=[35], line_dash_offset=30, **kwargs)
            p.renderers.extend([start_line1, start_line2])

    return disclaimer


def generate_pit_stop_plot(race_pit_stops, cached_driver_map, race_laps):
    """
    Generates a plot of when pit stops happened.
    :param race_pit_stops: Pit stop data for this race
    :param cached_driver_map: Cached driver map
    :param race_laps: Race laps
    :return: Figure layout
    """
    logging.info("Generating pit stop plot")

    if race_pit_stops.shape[0] == 0:
        return Div(text="Unfortunately, we do not have any pit stop timing data on this race. We have lap timing data "
                        "for the 2012 season onwards.")

    max_laps = race_laps["lap"].max()
    x = np.arange(1, max_laps + 1)
    data = {"laps": x}
    driver_names = []
    colors = []
    hatching = []

    for driver_id in race_laps["driverId"].unique():
        if driver_id not in cached_driver_map.keys():
            continue
        name = cached_driver_map[driver_id][0]
        color = cached_driver_map[driver_id][2]
        line_dash = cached_driver_map[driver_id][3]

        pit_stop_laps = race_pit_stops[race_pit_stops["driverId"] == driver_id]["lap"].values
        encoded = np.zeros(max_laps)
        for lap in pit_stop_laps:
            encoded[lap - 1] += 1
        data[name] = encoded
        driver_names.append(name)
        colors.append(color)
        hatching.append(" " if line_dash is "solid" else "/")

    data = ColumnDataSource(data=data)

    pit_stop_plot = figure(
        title=u"Pit stop plot \u2014 every time a driver pits",
        x_axis_label="Lap",
        y_axis_label="Lap Time",
        x_range=Range1d(0.5, max_laps + 1, bounds=(0.5, max_laps + 1)),
        plot_height=30 * len(cached_driver_map),
        tools="reset,save"
    )
    pit_stop_plot.xaxis.ticker = np.arange(0, 100, 5).tolist() + [1]
    pit_stop_plot.yaxis.ticker = np.arange(0, 50)

    renders = pit_stop_plot.vbar_stack(driver_names, x="laps", width=1, source=data, color=colors,
                                       hatch_pattern=hatching)

    # Generate the legend info
    legend = []
    for driver_id, glyph in zip(race_laps["driverId"].unique(), renders):
        if driver_id not in cached_driver_map:
            continue
        name = cached_driver_map[driver_id][0]
        legend_index = cached_driver_map[driver_id][1]
        legend.append(LegendItem(label=name, renderers=[glyph], index=legend_index))
    legend = sorted(legend, key=lambda l: l.index)
    legend = Legend(items=legend, location="top_right", glyph_height=15, spacing=2, inactive_fill_color="gray")
    pit_stop_plot.add_layout(legend, "right")
    pit_stop_plot.legend.click_policy = "mute"
    pit_stop_plot.legend.label_text_font_size = "10pt"  # The default font size

    pit_stop_plot.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Name", "$name")
    ]))

    return pit_stop_plot


def generate_lap_time_plot(race_laps, cached_driver_map, stdev_range=(1, 1), include_hist=True, highlight_dids=None,
                           muted_dids=None):
    """
    Generates a plot of lap times.
    :param race_laps: Race laps
    :param cached_driver_map: The map of driver info
    :param stdev_range: Used to determine y axis default range, defines min and max range in terms of standard
    deviations from the mean lap time
    :param include_hist: If True, will include the histogram of lap time distribution on the side, if False then won't
    :param highlight_dids: List of driver IDs to highlight
    :param muted_dids: List of driver IDs to have initially muted
    :return: The full lap time plot layout, the plot itself
    """
    if muted_dids is None:
        muted_dids = []
    if highlight_dids is None:
        highlight_dids = []
    logging.info("Generating lap time plot")

    if race_laps.shape[0] == 0:
        dummy = Spacer(width=0, height=0, background=PLOT_BACKGROUND_COLOR)
        return dummy, dummy

    # Get these values for y axis range
    mean_time = np.mean(race_laps["milliseconds"])
    stdev = np.std(race_laps["milliseconds"])
    no_outliers = race_laps[(race_laps["milliseconds"] - mean_time < 0.5 * stdev) &
                            (race_laps["milliseconds"] - mean_time > -0.5 * stdev)]
    if no_outliers.shape[0] / race_laps.shape[0] < 0.5:
        no_outliers = race_laps[(race_laps["milliseconds"] - mean_time < 1 * stdev) &
                                (race_laps["milliseconds"] - mean_time > -1 * stdev)]
    mean_time = np.mean(no_outliers["milliseconds"])
    stdev = np.std(no_outliers["milliseconds"])

    plot_kwargs = {
        "title": u"Lap time plot \u2014 Driver's lap time for every lap",
        "x_axis_label": "Lap",
        "y_axis_label": "Lap Time",
        "y_range": DataRange1d(start=pd.to_datetime(mean_time - stdev_range[0] * stdev, unit="ms"),
                               end=pd.to_datetime(mean_time + stdev_range[1] * stdev, unit="ms")),
        "tools": "pan,xbox_zoom,xwheel_zoom,ywheel_zoom,reset,box_zoom,wheel_zoom,save",
        "plot_height": 30 * len(cached_driver_map)
    }
    max_laps = race_laps["lap"].max()
    lap_time_plot = figure(
        title=u"Lap time plot \u2014 Driver's lap time for every lap",
        x_axis_label="Lap",
        y_axis_label="Lap Time",
        x_range=Range1d(1, max_laps, bounds=(1, max_laps + 50)),
        y_range=DataRange1d(start=pd.to_datetime(mean_time - stdev_range[0] * stdev, unit="ms"),
                            end=pd.to_datetime(mean_time + stdev_range[1] * stdev, unit="ms")),
        tools="pan,xbox_zoom,xwheel_zoom,reset,box_zoom,wheel_zoom,save",
        plot_height=30 * len(cached_driver_map)
    )
    lap_time_plot.xaxis.ticker = FixedTicker(ticks=[1] + list(np.arange(0, 500, 5)))

    # Set up the y axis
    lap_time_plot.yaxis.formatter = DatetimeTickFormatter(**DATETIME_TICK_KWARGS)

    legend = []
    for driver_id in race_laps["driverId"].unique():
        if driver_id not in cached_driver_map.keys():
            continue
        driver_laps = race_laps[race_laps["driverId"] == driver_id].sort_values("lap")
        x = driver_laps["lap"]
        y = driver_laps["milliseconds"]
        y = pd.to_datetime(y, unit="ms", errors="ignore")
        n = x.shape[0]

        # Use the cached info
        name = cached_driver_map[driver_id][0]
        legend_index = cached_driver_map[driver_id][1]
        color = cached_driver_map[driver_id][2]
        line_dash = cached_driver_map[driver_id][3]
        driver_positions = cached_driver_map[driver_id][4]
        finish_position = cached_driver_map[driver_id][5]
        constructor_name = cached_driver_map[driver_id][6]
        driver_lap_times = cached_driver_map[driver_id][7]

        if driver_positions.shape[0] > n:  # Pad to be proper length
            driver_positions = driver_positions[:n]
        elif driver_positions.shape[0] < n:
            max_lap = driver_laps["lap"].max()
            last_pos = driver_laps[driver_laps["lap"] == max_lap]["position"].values[0]
            driver_positions = driver_positions.append(pd.Series([str(last_pos)] * (n - driver_positions.shape[0])),
                                                       ignore_index=True)

        if driver_lap_times.shape[0] > n:  # Pad to be proper length
            driver_lap_times = driver_lap_times[:n]
        elif driver_lap_times.shape[0] < n:
            driver_lap_times = driver_lap_times.append(pd.Series([""] * (n - driver_lap_times.shape[0])),
                                                       ignore_index=True)

        source = ColumnDataSource(data=dict(x=x, y=y,
                                            name=[name] * n,
                                            positions=driver_positions,
                                            positions_str=driver_positions.apply(int_to_ordinal),
                                            finish_position=[finish_position] * n,
                                            finish_position_str=[int_to_ordinal(finish_position)] * n,
                                            constructor=[constructor_name] * n,
                                            lap_time=driver_lap_times))
        line_width = get_line_thickness(finish_position)
        if driver_id in highlight_dids:
            line_width *= 1.5
        elif driver_id not in highlight_dids and len(highlight_dids) > 0:
            line_width *= 0.5
        line = lap_time_plot.line(x="x", y="y", source=source, color=color, line_dash=line_dash,
                                  line_width=line_width, line_alpha=0.7, muted_alpha=0.01)
        if driver_id in muted_dids:
            line.muted = True

        legend_item = LegendItem(label=name, renderers=[line], index=legend_index)
        legend.append(legend_item)

    legend = sorted(legend, key=lambda l: l.index)
    legend = Legend(items=legend, location="top_right", glyph_height=15, spacing=2, inactive_fill_color="gray")
    lap_time_plot.add_layout(legend, "right")
    lap_time_plot.legend.click_policy = "mute"
    lap_time_plot.legend.label_text_font_size = "12pt"  # The default font size

    lap_time_plot.add_tools(CrosshairTool(dimensions="both", line_color="white", line_alpha=0.6))

    # Add the hover tooltip
    lap_time_plot.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Name", "@name"),
        ("Lap", "@x"),
        ("Current Position", "@positions_str"),
        ("Final Position", "@finish_position_str"),
        ("Lap Time", "@lap_time"),
        ("Constructor", "@constructor"),
    ]))

    y = race_laps["milliseconds"]
    vhist, vedges = np.histogram(y, bins=500)
    vmax = max(vhist) * 1.1

    pv = figure(
        toolbar_location=None,
        plot_width=200,
        plot_height=lap_time_plot.plot_height,
        x_axis_label="Count",
        x_range=Range1d(0, vmax, bounds=(0, vmax + 100)),
        y_range=lap_time_plot.y_range,
        min_border=10,
        y_axis_location="right",
        y_axis_type="datetime"
    )
    pv.xaxis.ticker.desired_num_ticks = 5
    pv.yaxis.formatter = DatetimeTickFormatter(**DATETIME_TICK_KWARGS)
    pv.ygrid.grid_line_color = None
    pv.xaxis.major_label_orientation = np.pi / 4
    pv.background_fill_color = "#0f0f0f"

    pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, line_color="#f0f0f0", line_width=2, fill_alpha=0)

    layout = [row(lap_time_plot, sizing_mode="stretch_width")]
    if include_hist:
        layout.append(pv)
    layout = row(layout)
    return layout, lap_time_plot


def generate_position_plot(race_laps, cached_driver_map, highlight_dids=None, muted_dids=None):
    """
    Generates a position vs. time plot.
    :param race_laps: Lap times for this race
    :param cached_driver_map: Map storing some info on each driver
    :param highlight_dids: Driver IDs to highlight
    :param muted_dids: Driver IDs to have initially muted
    :return: The position plot or a Spacer if there is an error
    """
    logging.info("Generating position plot")
    if muted_dids is None:
        muted_dids = []
    if highlight_dids is None:
        highlight_dids = []

    if race_laps.shape[0] == 0:
        return Spacer(width=0, height=0, background=PLOT_BACKGROUND_COLOR)

    max_laps = race_laps["lap"].max()
    position_plot = figure(
        title=u"Position plot \u2014 Driver's position over time",
        x_axis_label="Lap",
        y_axis_label="Position",
        x_range=Range1d(1, max_laps, bounds=(1, max_laps + 50)),
        y_range=Range1d(0, 22, bounds=(0, 60)),
        plot_height=30 * len(cached_driver_map)
    )
    position_plot.xaxis.ticker = FixedTicker(ticks=[1] + list(np.arange(10, 200, 10)))
    position_plot.yaxis.ticker = FixedTicker(ticks=[1] + list(np.arange(5, 60, 5)))

    legend = []
    for driver_id in race_laps["driverId"].unique():
        if driver_id not in cached_driver_map.keys():  # Did not qualify case
            continue
        driver_laps = race_laps[race_laps["driverId"] == driver_id].sort_values("lap")
        x = driver_laps["lap"]
        y = driver_laps["position"]
        n = x.shape[0]

        # Use the cached info
        name = cached_driver_map[driver_id][0]
        legend_index = cached_driver_map[driver_id][1]
        color = cached_driver_map[driver_id][2]
        line_dash = cached_driver_map[driver_id][3]
        driver_positions = cached_driver_map[driver_id][4]
        finish_position = cached_driver_map[driver_id][5]
        constructor_name = cached_driver_map[driver_id][6]

        if driver_positions.shape[0] > n:  # Pad to be proper length
            driver_positions = driver_positions[:n]
        elif driver_positions.shape[0] < n:
            max_lap = driver_laps["lap"]
            last_pos = driver_laps[driver_laps["lap"] == max_lap]["position"].values[0]
            driver_positions = driver_positions.append(pd.Series([str(last_pos)] * (n - driver_positions.shape[0])),
                                                       ignore_index=True)

        source = ColumnDataSource(data=dict(x=x, y=y,
                                            name=[name] * n,
                                            positions=driver_positions,
                                            positions_str=driver_positions.apply(int_to_ordinal),
                                            finish_position=[finish_position] * n,
                                            finish_position_str=[int_to_ordinal(finish_position)] * n,
                                            constructor=[constructor_name] * n))
        line_width = get_line_thickness(finish_position)
        if driver_id in highlight_dids:
            line_width *= 1.5
        elif driver_id not in highlight_dids and len(highlight_dids) > 0:
            line_width *= 0.5
        line = position_plot.line(x="x", y="y", source=source, color=color, line_dash=line_dash,
                                  line_width=line_width, line_alpha=0.7, muted_alpha=0.08)
        if driver_id in muted_dids:
            line.muted = True

        legend_item = LegendItem(label=name, renderers=[line], index=legend_index)
        legend.append(legend_item)

    legend = sorted(legend, key=lambda l: l.index)
    legend = Legend(items=legend, location="top_right", glyph_height=15, spacing=2, inactive_fill_color="gray")
    position_plot.add_layout(legend, "right")
    position_plot.legend.click_policy = "mute"
    position_plot.legend.label_text_font_size = "12pt"  # The default font size

    position_plot.add_tools(CrosshairTool(dimensions="both", line_color="white", line_alpha=0.6))

    # Add the hover tooltip
    position_plot.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Name", "@name"),
        ("Lap", "@x"),
        ("Current Position", "@positions_str"),
        ("Final Position", "@finish_position_str"),
        ("Constructor", "@constructor"),
    ]))

    return position_plot


def generate_gap_plot(race_laps, race_results, highlight_dids=None, muted_dids=None):
    """
    Generates the gap plot for the given race.
    :param race_laps: Slice of lap times for this race
    :param race_results: Slice of results for this race
    :param highlight_dids: Driver IDs to highlight
    :param muted_dids: Driver IDs to have initially muted
    :return: The full gap plot along with some cached info on each driver or an error div
    """
    logging.info("Generating gap plot")
    if muted_dids is None:
        muted_dids = []
    if highlight_dids is None:
        highlight_dids = []

    if race_laps.shape[0] == 0:
        return Div(text="Unfortunately, we do not have any lap timing data on this race. We have lap timing data for "
                        "the 1996 season onwards. If you know a source for this data, please contact us here."), {}

    # Get some useful slices
    winner_did = race_results[race_results["position"] == 1]["driverId"].values[0]
    winner_median_time = np.mean(race_laps[race_laps["driverId"] == winner_did]["milliseconds"])
    winner_times = race_laps[race_laps["driverId"] == winner_did]["milliseconds"].values

    # Calculate the actual gaps
    gaps = []
    dids = race_laps["driverId"].unique()
    for driver_id in dids:
        driver_laps = race_laps[race_laps["driverId"] == driver_id].sort_values(by="lap")
        winner_car_time = 0
        driver_time = 0
        lap = 1

        for time in driver_laps["milliseconds"]:
            # Keep track of the imaginary "average" winning driver
            driver_time += time
            winner_car_time += winner_median_time if lap - 1 >= winner_times.shape[0] else winner_times[lap - 1]

            gap = winner_car_time - driver_time
            gap /= 1000  # milliseconds to seconds
            gaps.append([driver_id, gap, lap])
            lap += 1

    gaps = pd.DataFrame(gaps, columns=["driverId", "gap", "lap"])

    max_laps = race_laps["lap"].max()
    gap_plot = figure(
        title=u"Gap plot \u2014 Gap to the leader's average pace",
        x_axis_label="Lap",
        y_axis_label="seconds",
        tools="pan,xbox_zoom,xwheel_zoom,reset,box_zoom,wheel_zoom,save",
        x_range=Range1d(1, max_laps, bounds=(1, max_laps + 50)),
        y_range=(-180, gaps["gap"].max()),  # Cut off the plot at gap of -180
        plot_height=30 * dids.shape[0]
    )
    gap_plot.add_tools(CrosshairTool(dimensions="both", line_color="white", line_alpha=0.6))
    gap_plot.xaxis.ticker = FixedTicker(ticks=[1] + list(np.arange(0, max_laps, 5)))
    gap_plot.renderers.extend([Span(location=0, line_color="white", dimension="width", line_alpha=0.5, line_width=2)])

    # We want the color scheme to be same team means same color, but the two drivers have solid vs. dashed lines
    cached_driver_map = {}  # driverId: legend entry
    color_dash_gen = ColorDashGenerator()
    legend = []
    for driver_id in gaps["driverId"].unique():
        # Get the plotting data
        x = gaps[gaps["driverId"] == driver_id]["lap"]
        y = gaps[gaps["driverId"] == driver_id]["gap"]
        n = x.shape[0]

        if race_results[race_results["driverId"] == driver_id].shape[0] == 0:  # Did not qualify case
            continue

        driver_laps = race_laps[race_laps["driverId"] == driver_id]

        # Get the lap time string
        driver_lap_times = driver_laps["milliseconds"]
        driver_lap_times = pd.to_datetime(driver_lap_times, unit="ms", errors="ignore")
        driver_lap_times = driver_lap_times.apply(lambda x: x.strftime("%-M:%S.%f")).str[:-3]
        if driver_lap_times.shape[0] > n:  # Pad to be proper length
            driver_lap_times = driver_lap_times[:n]
        elif driver_lap_times.shape[0] < n:
            driver_lap_times = driver_lap_times.append(pd.Series([""] * (n - driver_lap_times.shape[0] - 2)),
                                                       ignore_index=True)

        # Get the position string
        driver_positions = driver_laps["position"].astype(str)
        if driver_positions.shape[0] > n:  # Pad to be proper length
            driver_positions = driver_positions[:n]
        elif driver_positions.shape[0] < n:
            driver_positions = driver_positions.append(pd.Series([""] * (n - driver_positions.shape[0] - 2)),
                                                       ignore_index=True)

        # Get finish position string
        finish_position = race_results[race_results["driverId"] == driver_id]["position"]
        finish_position = finish_position.fillna(race_results["position"].max())
        if finish_position.shape[0] == 0:
            finish_position = str(int(race_results["position"].max()))
        else:
            finish_position = finish_position.map("{:,.0f}".format).astype(str).values[0]
        if finish_position == "" or finish_position == " " or finish_position.lower() == "nan":
            finish_position = "DNF"

        # Get name string
        name = get_driver_name(driver_id)

        # Get constructor name string
        constructor_id = race_results[race_results["driverId"] == driver_id]["constructorId"]
        if constructor_id.shape[0] == 0:
            constructor_id = -1
            constructor_name = ""
        else:
            constructor_id = constructor_id.values[0]
            constructor_name = get_constructor_name(constructor_id)

        # Make gap text string
        gap_text = y.apply(lambda x: f"{abs(x)} seconds" if x < 0 else f"{x} seconds ahead")

        # Get coloring and line dash
        color, line_dash = color_dash_gen.get_color_dash(driver_id, constructor_id)

        # Constructor the data source with all the info for tooltip
        source = ColumnDataSource(data=dict(x=x, y=y,
                                            gap=gap_text,
                                            lap_times=driver_lap_times,
                                            positions=driver_positions,
                                            positions_str=driver_positions.apply(int_to_ordinal),
                                            finish_position=[finish_position] * n,
                                            finish_position_str=[int_to_ordinal(finish_position)] * n,
                                            name=[name] * n,
                                            constructor=[constructor_name] * n))
        line_width = get_line_thickness(finish_position)
        if driver_id in highlight_dids:
            line_width *= 1.5
        elif driver_id not in highlight_dids and len(highlight_dids) > 0:
            line_width *= 0.5
        line = gap_plot.line(x="x", y="y", source=source, color=color, line_dash=line_dash,
                             line_width=line_width, line_alpha=0.7, muted_alpha=0.08)

        if driver_id in muted_dids:
            line.muted = True

        # The legend will be arranged by finishing position
        position = int(race_results[race_results["driverId"] == driver_id]["positionOrder"].values[0])
        legend_item = LegendItem(label=name, renderers=[line], index=position - 1)
        legend.append(legend_item)
        cached_driver_map[driver_id] = [name, position - 1, color, line_dash,  # drawing and hover stuff
                                        driver_positions, finish_position, constructor_name, driver_lap_times]

    # Make and place the legend
    legend = sorted(legend, key=lambda l: l.index)
    legend = Legend(items=legend, location="top_right", glyph_height=15, spacing=2, inactive_fill_color="gray")
    gap_plot.add_layout(legend, "right")
    gap_plot.legend.click_policy = "mute"
    gap_plot.legend.label_text_font_size = "12pt"  # The default font size
    # Add the hover tooltip
    gap_plot.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Name", "@name"),
        ("Lap", "@x"),
        ("Current Position", "@positions_str"),
        ("Final Position", "@finish_position_str"),
        ("Constructor", "@constructor"),
        ("Lap time", "@lap_times"),
        ("Gap to winner", "@gap"),
    ]))

    return gap_plot, cached_driver_map


def generate_spvfp_scatter(race_results, race, race_driver_standings):
    """
    Start position vs finish position scatter
    :param race_results: Race results
    :param race: Slice of races.csv containing this race and this race only
    :param race_driver_standings: Race driver standings
    :return: Plot layout
    """
    return driver.generate_spvfp_scatter(race_results, race, race_driver_standings, include_driver_name_labels=True,
                                         color_drivers=True)


def generate_mltr_fp_scatter(race_results, race, race_driver_standings):
    """
    Mean lap time rank vs finish position scatter
    :param race_results: Race results
    :param race: Slice of races.csv containing this race and this race only
    :param race_driver_standings: Race driver standings
    :return: Plot layout
    """
    return driver.generate_mltr_fp_scatter(race_results, race, race_driver_standings,
                                           include_driver_name_lables=True, color_drivers=True)


def generate_error_layout(year_id, circuit_id):
    """
    Generates an error layout in the event that there was no race in the given year at the given circuit.
    :param year_id: Year
    :param circuit_id: Circuit ID
    :return: Div layout
    """
    logging.info("Generating error layout")
    circuit_name = circuits.loc[circuit_id, "name"]
    years_circuit_held = races[races["circuitId"] == circuit_id].sort_values(by="year")["year"].values
    circuits_that_year = races[races["year"] == year_id]["circuitId"]
    circuits_that_year = circuits[circuits.index.isin(circuits_that_year)]["name"].values

    # Generate the text
    text = f"Unfortunately, {circuit_name} did not host a Grand Prix in {year_id}. Here are some other options:<br>"
    text += f"The {circuit_name} hosted Grand Prix in..."
    text += "<ul>"
    for year in years_circuit_held:
        text += f"<li>{str(year)}</li>"
    text += "</ul><br>"
    text += f"In {year_id}, Grand Prix were hosted at..."
    text += "<ul>"
    for circuit in circuits_that_year:
        text += f"<li>{circuit}</li>"
    text += "</ul>"

    layout = Div(text=text)

    return layout

