import logging
import math
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import Div, Spacer, Range1d, FixedTicker, TableColumn, DataTable, ColumnDataSource, CrosshairTool, \
    HoverTool, Slider, Legend, LegendItem, Span, Title, Label
from bokeh.plotting import figure
from pandas import Series
import numpy as np
from utils import get_constructor_name, PLOT_BACKGROUND_COLOR, get_race_name, int_to_ordinal, \
    get_status_classification, millis_to_str, get_driver_name, result_to_str
from data_loading.data_loader import load_results, load_constructor_standings, load_races, load_fastest_lap_data, \
    load_status, load_driver_standings
from mode import year, constructor, driver, driverconstructor

# Note, YC = year constructor

results = load_results()
constructor_standings = load_constructor_standings()
driver_standings = load_driver_standings()
races = load_races()
fastest_lap_data = load_fastest_lap_data()
status = load_status()


def get_layout(year_id=-1, constructor_id=-1, **kwargs):
    if year_id < 1958:
        return Div(text="The constructor's championship did not exist this year! It started in 1958.")
    year_races = races[races["year"] == year_id]
    year_results = results[results["raceId"].isin(year_races.index)]
    yc_results = year_results[year_results["constructorId"] == constructor_id]
    # Detect if it is a valid combo
    if yc_results.shape[0] == 0:
        return generate_error_layout(year_id, constructor_id)

    # Generate more slices
    year_constructor_standings = constructor_standings[constructor_standings["raceId"].isin(year_races.index)]
    year_constructor_standings = year_constructor_standings.sort_values(by="raceId")
    yc_constructor_standings = year_constructor_standings[year_constructor_standings["constructorId"] == constructor_id]
    year_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"].isin(year_races.index)]
    yc_fastest_lap_data_idxs = []
    yc_driver_standings_idxs = []
    for idx, results_row in yc_results.iterrows():
        rid = results_row["raceId"]
        did = results_row["driverId"]
        fl_slice = year_fastest_lap_data[(year_fastest_lap_data["raceId"] == rid) &
                                         (year_fastest_lap_data["driver_id"] == did)]
        yc_fastest_lap_data_idxs.extend(fl_slice.index.values.tolist())
        driver_standings_slice = driver_standings[(driver_standings["raceId"] == rid) &
                                                  (driver_standings["driverId"] == did)]
        yc_driver_standings_idxs.extend(driver_standings_slice.index.values.tolist())
    yc_fastest_lap_data = fastest_lap_data.loc[yc_fastest_lap_data_idxs]
    yc_driver_standings = driver_standings.loc[yc_driver_standings_idxs]
    yc_races = year_races[year_races.index.isin(yc_results["raceId"])]

    logging.info(f"Generating layout for mode YEARCONSTRUCTOR in yearconstructor, year_id={year_id}, "
                 f"constructor_id={constructor_id}")

    # WCC plot
    wcc_plot = generate_wcc_plot(year_races, year_constructor_standings, year_results, constructor_id)

    # Positions plot
    positions_plot, positions_source = generate_positions_plot(yc_constructor_standings, yc_results,
                                                               yc_fastest_lap_data, year_id, constructor_id)

    # Start pos v finish pos scatter
    spvfp_scatter = generate_spvfp_scatter(yc_results, yc_races, yc_driver_standings)

    # Mean lap time rank vs finish pos scatter
    mltr_fp_scatter = generate_mltr_fp_scatter(yc_results, yc_races, yc_driver_standings)

    # Win plot
    win_plot = generate_win_plot(positions_source, constructor_id)

    # Finish pos bar plot
    finishing_position_bar_plot = generate_finishing_position_bar_plot(yc_results)

    # Driver performance table
    driver_performance_layout = generate_driver_performance_table(yc_races, yc_results)

    # Results table
    results_table, results_source = generate_results_table(yc_results, yc_fastest_lap_data, year_results,
                                                           year_fastest_lap_data)

    # Teammate comparison line plot
    teammate_comparison_line_plot, comparison_source = generate_teammate_comparison_line_plot(yc_results, year_races,
                                                                                              yc_driver_standings,
                                                                                              year_id)

    # Stats layout
    stats_layout = generate_stats_layout(positions_source, yc_results, comparison_source, year_id, constructor_id)

    # Header
    constructor_name = get_constructor_name(constructor_id)
    # todo add notes about colors and stuff
    header = Div(text=f"<h2><b>What did {constructor_name}'s {year_id} season look like?</b></h2>")

    logging.info("Finished generating layout for mode YEARCONSTRUCTOR")

    middle_spacer = Spacer(width=5, background=PLOT_BACKGROUND_COLOR)
    layout = column([header,
                     wcc_plot, middle_spacer,
                     positions_plot, middle_spacer,
                     win_plot, middle_spacer,
                     finishing_position_bar_plot, middle_spacer,
                     row([spvfp_scatter, mltr_fp_scatter], sizing_mode="stretch_width"), middle_spacer,
                     teammate_comparison_line_plot, middle_spacer,
                     driver_performance_layout,
                     stats_layout,
                     results_table],
                    sizing_mode="stretch_width")
    return layout


def generate_wcc_plot(year_races, year_constructor_standings, year_results, constructor_id, consider_window=2):
    """
    Generates a plot of WCC progression, focusing on the 5 constructors around the constructor
    :param year_races: Year races
    :param year_constructor_standings: Year cosntructor standings
    :param year_results: Year results
    :param constructor_id: Constructor ID
    :param consider_window: Window to consider (e.g. if set to 2, will focus on 5 constructors)
    :return: WCC plot layout
    """
    # Get the driver's final position
    final_rid = year_races[year_races["round"] == year_races["round"].max()].index.values[0]
    final_standings = year_constructor_standings[year_constructor_standings["raceId"] == final_rid]
    final_standings = final_standings.set_index("constructorId")
    if constructor_id in final_standings.index:
        constructor_final_standing = final_standings.loc[constructor_id, "position"]
        if isinstance(constructor_final_standing, Series):
            constructor_final_standing = constructor_final_standing.values[0]
        if constructor_final_standing > consider_window:
            min_position = constructor_final_standing - consider_window
            max_position = constructor_final_standing + consider_window
        else:
            min_position = 1
            max_position = 2 * consider_window + 1
        considering_cids = final_standings[(final_standings["position"] >= min_position) &
                                           (final_standings["position"] <= max_position)].index.unique().tolist()
        all_cids = set(final_standings.index)
        muted_cids = all_cids - set(considering_cids)
        return year.generate_wcc_plot(year_constructor_standings, year_results, highlight_cid=constructor_id,
                                      muted_cids=muted_cids)
    constructor_name = get_constructor_name(constructor_id, include_flag=False)
    return Div(text=f"Unfortunately, we have encountered an error or {constructor_name} was never officially "
                    f"classified in this season.")


def generate_positions_plot(yc_constructor_standings, yc_results, yc_fastest_lap_data, year_id, constructor_id):
    """
    Generates a plot of WCC position (both rounds and full season), quali, fastest lap, and finishing position rank vs
    time all on the same graph.
    :return:
    :param yc_constructor_standings: YC constructor standings
    :param yc_results: YC results
    :param yc_fastest_lap_data: YC fastest lap data
    :param year_id: Year
    :param constructor_id: Constructor ID
    :return: Positions plot layout, positions source
    """
    constructor_years = np.array([year_id])
    kwargs = dict(
        return_components_and_source=True,
        smoothing_alpha=0.2,
        smoothing_muted=True,
        show_driver_changes=True,
    )
    positions_plot, positions_source = constructor.generate_positions_plot(constructor_years, yc_constructor_standings,
                                                                           yc_results, yc_fastest_lap_data,
                                                                           constructor_id, **kwargs)

    # Add the axis overrides
    x_min = positions_source["x"].min() - 0.001
    x_max = positions_source["x"].max() + 0.001
    positions_plot.x_range = Range1d(x_min, x_max, bounds=(x_min, x_max))
    positions_plot.xaxis.ticker = FixedTicker(ticks=positions_source["x"])
    positions_source["roundName"] = positions_source["roundName"].fillna("")
    positions_plot.xaxis.major_label_overrides = {row["x"]: row["roundName"] for idx, row in
                                                  positions_source.iterrows()}
    positions_plot.xaxis.major_label_orientation = 0.8 * math.pi / 2
    positions_plot.xaxis.axis_label = ""

    return positions_plot, positions_source


def generate_win_plot(positions_source, constructor_id):
    """
    Generate a plot of number of races, number of wins, number of podiums, number of points, win percent, podium
    percent, and points per race on one plot (3 axes).
    :param positions_source: Positions source
    :param constructor_id: Constructor ID
    :return: Win plot
    """
    # TODO add points and points per race to this plot
    win_plot = constructor.generate_win_plot(positions_source, constructor_id)

    # Override axes
    if not isinstance(win_plot, Div):
        x_min = positions_source["x"].min() - 0.001
        x_max = positions_source["x"].max() + 0.001
        win_plot.x_range = Range1d(x_min, x_max, bounds=(x_min, x_max))
        win_plot.xaxis.ticker = FixedTicker(ticks=positions_source["x"])
        win_plot.xaxis.major_label_overrides = {row["x"]: row["roundName"] for idx, row in
                                                positions_source.iterrows()}
        win_plot.xaxis.major_label_orientation = 0.8 * math.pi / 2
        win_plot.xaxis.axis_label = ""

    return win_plot


def generate_driver_performance_table(yc_races, yc_results):
    """
    Generates a table of drivers for this year and performance with each driver (number of wins, number of podiums,
    number of races, number of points, percentages for each, etc.)
    :param yc_races: YC races
    :param yc_results: YC results
    :return: Driver performance table layout
    """
    return constructor.generate_driver_performance_table(yc_races, yc_results)[0]


def generate_results_table(yc_results, yc_fastest_lap_data, year_results, year_fastest_lap_data, year_only=False,
                           height=None, include_driver_name=True, include_constructor_name=False):
    """
    Generates a table of results at each race, including quali position, finish position (or reason for DNF), time, gap
    to leader, fastest lap time and gap to fastest lap (of all drivers), average lap time and gap to fastest average lap
    time (of all drivers).
    :param yc_results: YC results
    :param yc_fastest_lap_data: YC fastest lap data
    :param year_results: Year results
    :param year_fastest_lap_data: Year fastest lap data
    :param year_only: Whether to set the race name row to just the year
    :param height: Plot height
    :param include_driver_name: If True, will include a driver name column
    :param include_constructor_name: If True, will include a constructor name column
    :return: Table layout, source
    """
    # TODO this might be able to be refactored with yeardriver or year, but it is kind of unique
    logging.info("Generating results table")
    source = pd.DataFrame(columns=["race_name", "driver_name", "driver_id ", "race_id", "year", "constructor_name",
                                   "quali_pos_str",
                                   "finish_pos_str",
                                   "time_str",
                                   "fastest_lap_time_str",
                                   "avg_lap_time_str"])
    for idx, results_row in yc_results.sort_values(by=["raceId", "driverId"]).iterrows():
        rid = results_row["raceId"]
        driver_id = results_row["driverId"]
        constructor_id = results_row["constructorId"]
        driver_name = get_driver_name(driver_id)
        constructor_name = get_constructor_name(constructor_id)
        race_results = year_results[year_results["raceId"] == rid]
        race_fastest_lap_data = year_fastest_lap_data[year_fastest_lap_data["raceId"] == rid]
        race_driver_fastest_lap_data = yc_fastest_lap_data[(yc_fastest_lap_data["raceId"] == rid) &
                                                           (yc_fastest_lap_data["driver_id"] == driver_id)]
        race_name = get_race_name(rid)
        grid = results_row["grid"]
        if grid == -1:
            quali_pos_str = "DNQ"
        else:
            quali_pos_str = int_to_ordinal(grid)
        status_id = results_row["statusId"]
        finish_pos_str, finish_pos = result_to_str(results_row["positionOrder"], status_id)
        time = results_row["milliseconds"]
        winner = race_results[race_results["positionOrder"] == 1]
        if winner.shape[0] > 0 and winner["driverId"].values[0] != driver_id \
                and not np.isnan(time) and not np.isnan(results_row["position"]):
            time_gap = millis_to_str(time - winner["milliseconds"].values[0])
            time_str = millis_to_str(time) + " (+" + time_gap + ")"
            if status_id != 1 and get_status_classification(status_id) == "finished":
                time_str = millis_to_str(time) + " (+" + time_gap + ", " + status.loc[status_id, "status"] + ")"
        elif finish_pos == 1:
            time_str = millis_to_str(time)
        else:
            time_str = "Not Set"
        if race_driver_fastest_lap_data.shape[0] > 0:
            fastest_lap_time = race_driver_fastest_lap_data["fastest_lap_time_millis"].values[0]
            fastest_lap_time_str = millis_to_str(fastest_lap_time)
            if race_driver_fastest_lap_data["rank"].values[0] == " 1":
                fastest_lap_time_str = fastest_lap_time_str + " (Fastest)"
            else:
                fastest_time = race_fastest_lap_data[race_fastest_lap_data["rank"] == " 1"]["fastest_lap_time_millis"]
                if fastest_time.shape[0] > 0 and not np.isnan(fastest_lap_time):
                    fastest_time = fastest_time.values[0]
                    fastest_gap = millis_to_str(fastest_lap_time - fastest_time)
                    fastest_lap_time_str = millis_to_str(fastest_lap_time) + " (+" + fastest_gap + ")"
            if fastest_lap_time_str == "":
                fastest_lap_time_str = "Not Set"
            fastest_avg_idx = race_fastest_lap_data["avg_lap_time_millis"].idxmin()
            avg_lap_time = race_driver_fastest_lap_data["avg_lap_time_millis"].values[0]
            if np.isnan(avg_lap_time):
                avg_lap_time_str = "Not Set"
            elif race_fastest_lap_data.loc[fastest_avg_idx, "driver_id"] == driver_id or np.isnan(avg_lap_time):
                avg_lap_time_str = millis_to_str(avg_lap_time) + " (Fastest Avg.)"
            else:
                fastest_avg_time = race_fastest_lap_data.loc[fastest_avg_idx, "avg_lap_time_millis"]
                avg_gap = millis_to_str(avg_lap_time - fastest_avg_time)
                avg_lap_time_str = millis_to_str(avg_lap_time) + " (+" + avg_gap + ")"
        else:
            fastest_lap_time_str = "Not Set"
            avg_lap_time_str = "Not Set"
        source = source.append({
            "race_name": race_name,
            "race_id": rid,
            "driver_name": driver_name,
            "driver_id": driver_id,
            "constructor_name": constructor_name,
            "year": races.loc[rid, "year"],
            "quali_pos_str": quali_pos_str,
            "finish_pos_str": finish_pos_str,
            "time_str": time_str,
            "fastest_lap_time_str": fastest_lap_time_str,
            "avg_lap_time_str": avg_lap_time_str
        }, ignore_index=True)

    if year_only:
        source = source.sort_values(by="race_name", ascending=False)

    results_columns = [
        TableColumn(field="quali_pos_str", title="Grid Pos.", width=75),
        TableColumn(field="finish_pos_str", title="Finish Pos.", width=75),
        TableColumn(field="time_str", title="Time", width=100),
        TableColumn(field="fastest_lap_time_str", title="Fastest Lap Time", width=75),
        TableColumn(field="avg_lap_time_str", title="Avg. Lap Time", width=75),
    ]
    if include_driver_name:
        results_columns.insert(0, TableColumn(field="driver_name", title="Driver Name", width=100))
    if include_constructor_name:
        results_columns.insert(0, TableColumn(field="constructor_name", title="Constructor Name", width=100))
    if year_only:
        results_columns.insert(0, TableColumn(field="year", title="Year", width=50))
    else:
        results_columns.insert(0, TableColumn(field="race_name", title="Race Name", width=100))
    results_table = DataTable(source=ColumnDataSource(data=source), columns=results_columns, index_position=None,
                              height=28 * yc_results.shape[0] if height is None else height)
    title = Div(text=f"<h2><b>Results for each race</b></h2><br><i>The fastest lap time and average lap time gaps "
                     f"shown are calculated based on the gap to the fastest of all drivers and fastest average of "
                     f"all drivers in that race respectively.</i>")
    return column([title, row([results_table], sizing_mode="stretch_width")], sizing_mode="stretch_width"), source


def generate_finishing_position_bar_plot(yc_results):
    """
    Generates a bar plot of race finishing positions
    :param yc_results: YC results
    :return: Finishing position bar plot
    """
    return constructor.generate_finishing_positions_bar_plot(yc_results)


def generate_spvfp_scatter(yc_results, yc_races, yc_driver_standings):
    """
    Generates a plot of start position vs finish position scatter plot (see driver.generate_spvfp_scatter)
    :param yc_results: YC results
    :param yc_races: YC races
    :param yc_driver_standings: YC driver standings
    :return: Start pos vs finish pos scatter
    """
    kwargs = dict(
        include_year_labels=False,
        include_race_labels=True,
        color_drivers=True
    )
    return driver.generate_spvfp_scatter(yc_results, yc_races, yc_driver_standings, **kwargs)


def generate_mltr_fp_scatter(yc_results, yc_races, yc_driver_standings):
    """
    Generates a plot of mean lap time rank vs finish position scatter plot (see circuitdriver.generate_mltr_fp_scatter).
    :param yc_results: YC results
    :param yc_races: YC races
    :param yc_driver_standings: YC driver standings
    :return: Mean lap time rank vs finish pos scatter
    """
    kwargs = dict(
        include_year_labels=False,
        include_race_labels=True,
        color_drivers=True
    )
    return driver.generate_mltr_fp_scatter(yc_results, yc_races, yc_driver_standings, **kwargs)


def generate_teammate_comparison_line_plot(yc_results, year_races, yc_driver_standings, year_id):
    """
    Generates a plot of finish position along with teammate finish position vs time, see
    driverconstructor.generate_teammate_comparison_line_plot.
    :param yc_results: YC results
    :param year_races: Year races
    :param yc_driver_standings: YC driver standings
    :param year_id: Year
    :return: Layout, source
    """
    kwargs = dict(smoothed_muted=False,
                  return_components_and_source=True,
                  highlight_driver_changes=True)
    slider, teammate_fp_plot, source = constructor.generate_teammate_comparison_line_plot(
        yc_results, year_races, yc_driver_standings, np.array([year_id]), **kwargs)
    teammate_fp_plot.xaxis.ticker = FixedTicker(ticks=source["x"])
    teammate_fp_plot.xaxis.major_label_overrides = {row["x"]: row["roundName"] for idx, row in source.iterrows()}
    teammate_fp_plot.xaxis.major_label_orientation = 0.8 * math.pi / 2
    teammate_fp_plot.xaxis.axis_label = ""
    return column([slider, teammate_fp_plot], sizing_mode="stretch_width"), source


def generate_stats_layout(positions_source, yc_results, comparison_source, year_id, constructor_id):
    """
    Year summary div, including WCC place, highest race finish, number of races, points, points per race, number of
    wins, number of podiums, and everything else in constructor.generate_stats_layout and
    yeardriver.generate_stats_layout
    - WCC place
    - Highest race finish
    - Number of races
    - Points
    - Points per race
    - Number of wins and where were they
    - Number of podiums and where were they
    - Teammates
    - Constructors
    - Mean gap to teammate in positions
    - Mean grid position
    - Mean finish position
    - DNF info
    :param positions_source: Positions source
    :param yc_results: YC results
    :param comparison_source: Comparison source
    :param year_id: Year ID
    :param constructor_id: Constructor ID
    :return: Stats layout
    """
    logging.info("Generating year constructor stats layout")
    if positions_source.shape[0] == 0:
        return Div(text="")
    wcc_final_standing = positions_source["wcc_final_standing"].mode()
    if wcc_final_standing.shape[0] > 0:
        wcc_final_standing_str = int_to_ordinal(wcc_final_standing.values[0])
    else:
        wcc_final_standing_str = ""
    highest_race_finish_idx = yc_results["positionOrder"].idxmin()
    if np.isnan(highest_race_finish_idx):
        highest_race_finish_str = ""
    else:
        highest_race_finish = yc_results.loc[highest_race_finish_idx, "positionOrder"]
        round_name = get_race_name(yc_results.loc[highest_race_finish_idx, "raceId"])
        highest_race_finish_str = int_to_ordinal(highest_race_finish) + " at " + round_name
    num_races = positions_source["race_id"].unique().shape[0]
    num_races_str = str(num_races)
    points = positions_source["points"].max()
    if np.isnan(points):
        points_str = ""
    elif points <= 0:
        points_str = str(points) + " (0 pts/race)"
    else:
        points_str = str(points) + " (" + str(round(points / num_races, 1)) + " pts/race)"
    wins_slice = yc_results[yc_results["positionOrder"] == 1]
    num_wins = wins_slice.shape[0]
    if num_wins == 0:
        wins_str = str(num_wins)
    else:
        wins_str = str(num_wins) + " (" + ", ".join(wins_slice["raceId"].apply(get_race_name)) + ")"
        if len(wins_str) > 120:
            split = wins_str.split(" ")
            split.insert(int(len(split) / 2), "<br>    " + "".ljust(20))
            wins_str = " ".join(split)
    podiums_slice = yc_results[yc_results["positionOrder"] <= 3]
    num_podiums = podiums_slice.shape[0]
    if num_podiums == 0:
        podiums_str = str(num_podiums)
    else:
        race_names = ", ".join([get_race_name(rid) for rid in podiums_slice["raceId"].unique()])
        podiums_str = str(num_podiums) + " (" + race_names + ")"
        if len(podiums_str) > 120:
            split = podiums_str.split(" ")
            split.insert(int(len(split) / 2), "<br>    " + "".ljust(20))
            podiums_str = " ".join(split)
    driver_dids = yc_results["driverId"].unique()
    driver_names = []
    for did in driver_dids:
        driver_names.append(get_driver_name(did))
    driver_names = ", ".join(driver_names)
    mean_grid_pos = yc_results["grid"].replace("", np.nan).mean()
    if np.isnan(mean_grid_pos):
        mean_grid_pos_str = ""
    else:
        mean_grid_pos_str = str(round(mean_grid_pos, 1))
    mean_finish_pos = yc_results["positionOrder"].mean()
    if np.isnan(mean_finish_pos):
        mean_finish_pos_str = ""
    else:
        mean_finish_pos_str = str(round(mean_finish_pos, 1))
    classifications = yc_results["statusId"].apply(get_status_classification)
    num_mechanical_dnfs = classifications[classifications == "mechanical"].shape[0]
    num_crash_dnfs = classifications[classifications == "crash"].shape[0]
    if num_races > 0:
        num_mechanical_dnfs_str = str(num_mechanical_dnfs) + " (" + \
                                  str(round(100 * num_mechanical_dnfs / num_races, 1)) + "%)"
        num_crash_dnfs_str = str(num_crash_dnfs) + " (" + str(round(100 * num_crash_dnfs / num_races, 1)) + "%)"
    else:
        num_mechanical_dnfs_str = ""
        num_crash_dnfs_str = ""
    mean_teammate_gap_pos = (comparison_source["driver1_fp"] - comparison_source["driver2_fp"]).mean()
    if np.isnan(mean_teammate_gap_pos):
        mean_teammate_gap_pos_str = ""
    else:
        mean_teammate_gap_pos_str = "Driver {} finished {} places better than driver {} on average"
        mean_teammate_gap_pos_str = mean_teammate_gap_pos_str.format("1" if mean_teammate_gap_pos < 0 else "2",
                                                                     str(abs(round(mean_teammate_gap_pos, 1))),
                                                                     "2" if mean_teammate_gap_pos < 0 else "1")

    # Construct the HTML
    header_template = """
    <h2 style="text-align: left;"><b>{}</b></h2>
    """
    template = """
    <pre><b>{}</b> {}<br></pre>
    """

    constructor_name = get_constructor_name(constructor_id, include_flag=False)
    constructor_stats = header_template.format(f"{constructor_name}'s Stats for the {year_id} Season")
    constructor_stats += template.format("WCC Final Pos.: ".ljust(20), wcc_final_standing_str)
    constructor_stats += template.format("Num. Races: ".ljust(20), num_races_str)
    if num_wins == 0:
        constructor_stats += template.format("Best Finish Pos.: ".ljust(20), highest_race_finish_str)
    constructor_stats += template.format("Wins: ".ljust(20), wins_str)
    constructor_stats += template.format("Podiums: ".ljust(20), podiums_str)
    constructor_stats += template.format("Points: ".ljust(20), points_str)
    constructor_stats += template.format("Drivers(s): ".ljust(20), driver_names)
    constructor_stats += template.format("Avg. Grid Pos.: ".ljust(20), mean_grid_pos_str)
    constructor_stats += template.format("Avg. Finish Pos.: ".ljust(20), mean_finish_pos_str)
    constructor_stats += template.format("Mechanical DNFs: ".ljust(20), num_mechanical_dnfs_str)
    constructor_stats += template.format("Crash DNFs: ".ljust(20), num_crash_dnfs_str)
    constructor_stats += template.format("Avg. Driver Gap: ".ljust(20), mean_teammate_gap_pos_str)

    return Div(text=constructor_stats)


def generate_error_layout(year_id, constructor_id):
    """
    Generates an error layout in the event that the given constructor never competed in the given year.
    :param year_id: Year
    :param constructor_id: Driver ID
    :return: Div layout
    """
    logging.info("Generating error layout")
    constructor_name = get_constructor_name(constructor_id, include_flag=False)
    constructor_results = results[results["constructorId"] == constructor_id]
    years_for_that_constructor = sorted(races.loc[constructor_results["raceId"].values.tolist(), "year"].unique())
    year_races = races[races["year"] == year_id]
    constructors_for_that_year = results[results["raceId"].isin(year_races.index.values)]["constructorId"].unique()

    # Generate the text
    text = f"Unfortunately, {constructor_name} never competed in the {year_id} season. Here are some other options:<br>"
    text += f"The following constructors competed in the {year_id} season..."
    text += "<ul>"
    for cid in constructors_for_that_year:
        text += f"<li>{get_constructor_name(cid)}</li>"
    text += "</ul>"
    text += f"{constructor_name} has competed in the following years..."
    text += "<ul>"
    for year in years_for_that_constructor:
        text += f"<li>{year}</li>"
    text += "</ul><br>"

    layout = Div(text=text)
    return layout


def is_valid_input(year_id, constructor_id):
    """
    Returns True if the given combination of year and constructor are valid
    :param year_id: Year ID
    :param constructor_id: Constructor ID
    :return: True if valid, False otherwise
    """
    if year_id < 1958:
        return False
    year_races = races[races["year"] == year_id]
    year_results = results[results["raceId"].isin(year_races.index)]
    yc_results = year_results[year_results["constructorId"] == constructor_id]
    return yc_results.shape[0] > 0

