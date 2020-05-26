import logging
import math
from datetime import datetime
from collections import defaultdict
import bokeh
import pandas as pd
import numpy as np
from bokeh.colors import RGB
from bokeh.layouts import column, row
from bokeh.models import Div, ColumnDataSource, Spacer, Range1d, LegendItem, Legend, HoverTool, FixedTicker, \
    CrosshairTool, Title, LabelSet, Label, Span
from bokeh.plotting import figure
from html_table import HTML_table
from html_table.HTML_table import TableCell, Table, TableRow
from data_loading.data_loader import load_seasons, load_driver_standings, load_races, load_results, \
    load_constructor_standings, load_lap_times, load_qualifying, load_constructor_mean_lap_times, load_fastest_lap_data
from utils import PLOT_BACKGROUND_COLOR, get_line_thickness, get_driver_name, get_constructor_name, \
    ColorDashGenerator, get_race_name, position_text_to_str, get_status_classification, rounds_to_str

seasons = load_seasons()
driver_standings = load_driver_standings()
constructor_standings = load_constructor_standings()
races = load_races()
results = load_results()
lap_times = load_lap_times()
qualifying = load_qualifying()
constructor_mean_lap_times = load_constructor_mean_lap_times()
fastest_lap_data = load_fastest_lap_data()


def get_layout(year_id=-1, **kwargs):
    if year_id not in seasons.index.unique():
        return generate_error_layout()

    # Generate some useful slices
    year_races = races[races["year"] == year_id]
    year_driver_standings = driver_standings[driver_standings["raceId"].isin(year_races.index)].sort_values(by="raceId")
    year_constructor_standings = constructor_standings[constructor_standings["raceId"].isin(year_races.index)]\
        .sort_values(by="raceId")
    year_results = results[results["raceId"].isin(year_races.index)]
    year_laps = lap_times[lap_times["raceId"].isin(year_races.index)]
    year_qualifying = qualifying[qualifying["raceId"].isin(year_races.index)]
    year_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"].isin(year_races.index)]

    logging.info(f"Generating layout for mode YEAR in year, year_id={year_id}")

    header = Div(text=f"<h2>What did the {year_id} season look like?</h2>")

    # Generate WDC plot
    wdc_plot = generate_wdc_plot(year_driver_standings, year_results)

    # Generate constructor's plot
    constructors_plot = generate_wcc_plot(year_constructor_standings, year_results)

    # Generate position vs mean lap time rank plot
    position_mlt_scatter = generate_position_mlt_scatter(year_laps, year_results,
                                                         year_driver_standings, year_constructor_standings)

    # Generate position vs mean finish minus mean start plot
    position_mfms_scatter = generate_position_mfms_scatter(year_results, year_driver_standings)

    # Generate the teams and drivers table
    teams_and_drivers = generate_teams_and_drivers(year_results, year_races)

    # Generate races info
    races_info = generate_races_info(year_races, year_qualifying, year_results, year_fastest_lap_data)

    # Generate WDC table
    wdc_table = generate_wdc_results(year_results, year_driver_standings, year_races)

    # Generate DNF table
    dnf_table = generate_dnf_table(year_results)

    def update_axis_sharing():  # TODO should I have axis sharing for this mode?
        pass

    # Bring it all together
    middle_spacer = Spacer(width=5, background=PLOT_BACKGROUND_COLOR)

    plots = column([wdc_plot, middle_spacer, constructors_plot, middle_spacer,
                    position_mlt_scatter, middle_spacer, position_mfms_scatter, middle_spacer],
                   sizing_mode="stretch_width")

    layout = column([header,
                     plots,
                     middle_spacer,
                     teams_and_drivers,
                     races_info,
                     wdc_table,
                     dnf_table],
                    sizing_mode="stretch_width")

    logging.info("Finished generating layout for mode YEAR")

    return layout


def generate_wdc_plot(year_driver_standings, year_results, highlight_did=None, muted_dids=None):
    """
    Generates a plot of the progress of the world driver's championship.
    :param year_driver_standings: Driver's championship standings for this year
    :param year_results: Results for this year
    :param highlight_did: Driver ID of a driver to be highlighted, leave to None if no highlight
    :param muted_dids: Driver IDs of drivers who should be initially muted
    :return: Plot layout
    """
    logging.info("Generating WDC plot")

    if muted_dids is None:
        muted_dids = []

    max_pts = year_driver_standings["points"].max()
    driver_ids = year_driver_standings["driverId"].unique()
    num_drivers = len(driver_ids)
    plot_height = 30 * min(num_drivers, 30)
    wdc_plot = figure(
        title=u"World Driver's Championship \u2014 Number of points each driver has",
        y_axis_label="Points",
        y_range=Range1d(0, max_pts + 5, bounds=(0, max_pts + 5)),
        tools="pan,box_zoom,wheel_zoom,reset,save",
        plot_height=plot_height
    )

    final_rid = year_driver_standings["raceId"].max()
    final_standings = year_driver_standings[year_driver_standings["raceId"] == final_rid]

    legend = []
    color_dash_gen = ColorDashGenerator()
    for driver_id in driver_ids:
        driver_standings = year_driver_standings[year_driver_standings["driverId"] == driver_id].copy()
        name = get_driver_name(driver_standings["driverId"].values[0])
        driver_standings["name"] = name

        final_standing = final_standings[final_standings["driverId"] == driver_id]["position"]
        if final_standing.shape[0] == 0:
            continue
        final_standing = final_standing.values[0]
        driver_standings["final_position"] = final_standing

        res = year_results[year_results["driverId"] == driver_id]
        default_constructor = res["constructorId"].mode().values[0]

        def get_constructor(rid):
            cid = res[res["raceId"] == rid]["constructorId"]
            shape = cid.shape
            cid = default_constructor if cid.shape[0] == 0 else cid.values[0]
            cid = -1 if np.isnan(cid).sum() == shape[0] else cid
            return get_constructor_name(cid)
        driver_standings["constructor_name"] = driver_standings["raceId"].apply(get_constructor)

        # Get the color and line dash
        color, line_dash = color_dash_gen.get_color_dash(driver_id, default_constructor)

        line_width = get_line_thickness(final_standing)
        if driver_id == highlight_did:
            line_width *= 1.5
            alpha = 0.9
        else:
            alpha = 0.68
        source = ColumnDataSource(data=driver_standings)
        line = wdc_plot.line(x="roundNum", y="points", source=source, line_width=line_width, color=color,
                             line_dash=line_dash, line_alpha=alpha, muted_alpha=0.05)

        legend_item = LegendItem(label=name, renderers=[line], index=final_standing - 1)
        legend.append(legend_item)

        if driver_id in muted_dids:
            line.muted = True

    # Format axes
    num_rounds = year_driver_standings["roundNum"].max()
    wdc_plot.x_range = Range1d(1, num_rounds + 0.01, bounds=(1, num_rounds))
    wdc_plot.xaxis.major_label_overrides = {row["roundNum"]: row["roundName"] for idx, row in
                                            year_driver_standings.iterrows()}
    wdc_plot.xaxis.ticker = FixedTicker(ticks=np.arange(1, num_rounds + 1))
    wdc_plot.xaxis.major_label_orientation = 0.8 * math.pi / 2

    # Legend
    legend = sorted(legend, key=lambda l: l.index)
    legend = Legend(items=legend, location="top_right", glyph_height=15, spacing=2, inactive_fill_color="gray")
    wdc_plot.add_layout(legend, "right")
    wdc_plot.legend.click_policy = "mute"
    wdc_plot.legend.label_text_font_size = "12pt"

    # Hover tooltip
    wdc_plot.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Name", "@name"),
        ("Round", "@roundNum - @roundName"),
        ("Points", "@points"),
        ("Current Position", "@position"),
        ("Final Position", "@final_position"),
        ("Constructor", "@constructor_name")
    ]))

    # Crosshair
    wdc_plot.add_tools(CrosshairTool(dimensions="height", line_color="white", line_alpha=0.6))

    return wdc_plot


def generate_wcc_plot(year_constructor_standings, year_results, highlight_cid=None, muted_cids=None):
    """
    Generates a plot of the progress of the constructor's championship.
    :param year_constructor_standings: Constructors's championship standings for this year
    :param year_results: Results for this year
    :param highlight_cid: Constructor ID of a constructor to be highlighted, leave to None if no highlight
    :param muted_cids: Constructor IDs of constructors who should be initially muted
    :return: Plot layout
    """
    if muted_cids is None:
        muted_cids = []
    logging.info("Generating WCC plot")
    if year_constructor_standings.shape[0] == 0:
        return Div(text="The constructor's championship did not exist this year! It started in 1958.")

    max_pts = year_constructor_standings["points"].max()
    constructor_ids = year_constructor_standings["constructorId"].unique()
    num_constructors = len(constructor_ids)
    constructors_plot = figure(
        title=u"World Constructors's Championship \u2014 Number of points each constructor has",
        y_axis_label="Points",
        y_range=Range1d(0, max_pts + 5, bounds=(0, max_pts + 5)),
        tools="pan,box_zoom,wheel_zoom,reset,save",
        plot_height=50 * min(num_constructors, 30)
    )

    final_rid = year_constructor_standings["raceId"].max()
    final_standings = year_constructor_standings[year_constructor_standings["raceId"] == final_rid]

    legend = []
    color_dash_gen = ColorDashGenerator()
    for constructor_id in constructor_ids:
        constructor_standings = year_constructor_standings[year_constructor_standings["constructorId"] == constructor_id]
        constructor_standings = constructor_standings.copy()
        name = get_constructor_name(constructor_standings["constructorId"].values[0])
        constructor_standings["name"] = name

        final_standing = final_standings[final_standings["constructorId"] == constructor_id]["position"].values[0]
        constructor_standings["final_position"] = final_standing

        res = year_results[year_results["constructorId"] == constructor_id]
        default_constructor = res["constructorId"].mode().values[0]

        # Get the color and line dash
        color, line_dash = color_dash_gen.get_color_dash(did=None, cid=default_constructor)

        line_width = get_line_thickness(final_standing)
        source = ColumnDataSource(data=constructor_standings)
        if constructor_id == highlight_cid:
            line_width *= 1.5
            alpha = 0.9
        else:
            alpha = 0.68
        line = constructors_plot.line(x="roundNum", y="points", source=source, line_width=line_width, color=color,
                                      line_dash=line_dash, line_alpha=alpha, muted_alpha=0.05)

        legend_item = LegendItem(label=name, renderers=[line], index=final_standing - 1)
        legend.append(legend_item)

        if constructor_id in muted_cids:
            line.muted = True

    # Format axes
    num_rounds = year_constructor_standings["roundNum"].max()
    constructors_plot.x_range = Range1d(1, num_rounds + 0.01, bounds=(1, num_rounds))
    constructors_plot.xaxis.major_label_overrides = {row["roundNum"]: row["roundName"] for idx, row in
                                                     year_constructor_standings.iterrows()}
    constructors_plot.xaxis.ticker = FixedTicker(ticks=np.arange(1, num_rounds + 1))
    constructors_plot.xaxis.major_label_orientation = 0.8 * math.pi / 2

    # Legend
    legend = sorted(legend, key=lambda l: l.index)
    legend = Legend(items=legend, location="top_right", glyph_height=15, spacing=2, inactive_fill_color="gray")
    constructors_plot.add_layout(legend, "right")
    constructors_plot.legend.click_policy = "mute"
    constructors_plot.legend.label_text_font_size = "12pt"

    # Hover tooltip
    constructors_plot.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Name", "@name"),
        ("Round", "@roundNum - @roundName"),
        ("Points", "@points"),
        ("Current Position", "@position"),
        ("Final Position", "@final_position")
    ]))

    # Crosshair
    constructors_plot.add_tools(CrosshairTool(dimensions="height", line_color="white", line_alpha=0.6))

    return constructors_plot


def generate_position_mlt_scatter(year_laps, year_results, year_driver_standings, year_constructor_standings):
    """
    Driver finish position vs their constructor's mean lap time rank to get a sense of who did well despite a worse
    car
    :param year_laps: Year laps
    :param year_results: Year results
    :param year_driver_standings: Year driver standings
    :param year_constructor_standings: Year constructor standings
    :return: Position vs mean lap time plot
    """
    # If we don't have the data, show nothing
    if year_laps.shape[0] == 0:
        return Spacer()
    logging.info("Generating position vs mean lap time scatter")

    final_rid = year_results["raceId"].max()
    final_driver_standings = year_driver_standings[year_driver_standings["raceId"] == final_rid]
    final_driver_standings = final_driver_standings.set_index("driverId")
    num_drivers = final_driver_standings.shape[0]
    final_constructor_standings = year_constructor_standings[year_constructor_standings["raceId"] == final_rid]
    final_constructor_standings = final_constructor_standings.set_index("constructorId")

    source = pd.DataFrame(columns=["short_name", "full_name", "constructor_name", "driver_final_standing",
                                   "constructor_final_standing", "constructor_mean_rank", "color"])
    position_mlt_scatter = figure(
        title=u"World Driver's Championship Position versus Constructor Mean Lap Time Rank \u2014 "
              u"Who did well with a poor car?",
        x_axis_label="World Driver's Championship Final Position",
        y_axis_label="Constructor Mean Lap Time Rank",
        tools="pan,reset,save",
        plot_height=30 * min(num_drivers, 30)
    )
    position_mlt_scatter.xaxis.ticker = FixedTicker(ticks=np.arange(5, 100, 5).tolist() + [1])

    subtitle = "The y axis is computed by finding the average lap time of every constructor at every race, and then " \
               "for every race, ranking each constructor based on mean lap time. Those ranks are then averaged."
    position_mlt_scatter.add_layout(Title(text=subtitle, text_font_style="italic"), "above")

    color_gen = ColorDashGenerator()
    for driver_id in year_results["driverId"].unique():
        driver_results = year_results[year_results["driverId"] == driver_id]
        full_name = get_driver_name(driver_id)
        short_name = get_driver_name(driver_id, include_flag=False, just_last=True)
        cid = driver_results["constructorId"].mode().values[0]
        constructor_name = get_constructor_name(cid)
        constructor_mean_rank = constructor_mean_lap_times[constructor_mean_lap_times["constructorId"] == cid]["rank"].mean()
        if driver_id in final_driver_standings.index:
            driver_final_standing = final_driver_standings.loc[driver_id, "position"]
        elif driver_id in year_driver_standings["driverId"]:
            driver_final_standing = year_driver_standings[year_driver_standings["driverId"] == driver_id]["position"]
            driver_final_standing = driver_final_standing[-1]
        else:
            continue
        constructor_final_standing = final_constructor_standings.loc[cid, "position"]

        color, _ = color_gen.get_color_dash(driver_id, cid)

        source = source.append({
            "short_name": short_name,
            "full_name": full_name,
            "constructor_name": constructor_name,
            "driver_final_standing": driver_final_standing,
            "constructor_final_standing": constructor_final_standing,
            "constructor_mean_rank": constructor_mean_rank,
            "color": color
        }, ignore_index=True)

    position_mlt_scatter.scatter(x="driver_final_standing", y="constructor_mean_rank", source=source, size=8,
                                 color="color")

    # Labels
    labels = LabelSet(x="driver_final_standing", y="constructor_mean_rank", text="short_name", level="glyph",
                      x_offset=0.7, y_offset=0.7, source=ColumnDataSource(data=source.to_dict(orient="list")),
                      render_mode="canvas", text_color="white", text_font_size="10pt", angle=0.2 * math.pi / 2)
    position_mlt_scatter.add_layout(labels)

    # Adjust axes a bit
    position_mlt_scatter.x_range = Range1d(0, 1.1 * source["driver_final_standing"].max())
    position_mlt_scatter.y_range = Range1d(0.8 * source["constructor_mean_rank"].min(),
                                           1.1 * source["constructor_mean_rank"].max())

    # Hover tooltip
    position_mlt_scatter.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Name", "@full_name"),
        ("Constructor", "@constructor_name"),
        ("Final Standing", "@driver_final_standing"),
        ("Constructor Final Standing", "@constructor_final_standing"),
        ("Constructor Mean Lap Time Rank", "@constructor_mean_rank")
    ]))

    # Crosshair
    position_mlt_scatter.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

    return position_mlt_scatter


def generate_position_mfms_scatter(year_results, year_driver_standings):
    """
    Generates a WDC position place vs mean of finishing position minus starting position plot to try to show how many
    places on average a driver makes up
    :param year_results: Year results
    :param year_driver_standings: Year driver's standings
    :return: Position vs mean finishing position minus mean start
    """
    final_rid = year_results["raceId"].max()
    final_driver_standings = year_driver_standings[year_driver_standings["raceId"] == final_rid]
    final_driver_standings = final_driver_standings.set_index("driverId")
    num_drivers = final_driver_standings.shape[0]

    source = pd.DataFrame(columns=["short_name", "full_name", "constructor_name", "driver_final_standing",
                                   "mean_spmfp", "color"])

    color_gen = ColorDashGenerator()
    for driver_id in final_driver_standings.index.unique():
        driver_results = year_results[year_results["driverId"] == driver_id]
        full_name = get_driver_name(driver_id)
        short_name = get_driver_name(driver_id, include_flag=False, just_last=True)
        cid = driver_results["constructorId"].mode().values[0]
        constructor_name = get_constructor_name(cid)
        final_standing = final_driver_standings.loc[driver_id, "position"]
        color, _ = color_gen.get_color_dash(driver_id, cid)

        start_position = driver_results["grid"]

        def get_adjusted_position(row):
            if np.isnan(row["position"]):
                rid = row["raceId"]
                return year_results[year_results["raceId"] == rid]["position"].max()
            else:
                return row["position"]
        finish_position = driver_results.apply(get_adjusted_position, axis=1)
        spmfp = start_position - finish_position

        source = source.append({
            "short_name": short_name,
            "full_name": full_name,
            "constructor_name": constructor_name,
            "driver_final_standing": final_standing,
            "mean_spmfp": spmfp.mean(),
            "color": color
        }, ignore_index=True)

    mean_spmfp_scatter = figure(
        title="World Driver's Championship Position versus Mean Starting Position minus Finish Position \u2014 "
              "Who fought their way up the order?",
        x_axis_label="World Driver's Championship Final Position",
        y_axis_label="Mean Start Position minus Finish Position",
        tools="pan,reset,save",
        plot_height=30 * min(num_drivers, 30)
    )
    mean_spmfp_scatter.xaxis.ticker = FixedTicker(ticks=np.arange(5, 100, 5).tolist() + [1])

    subtitle = "The y axis is computed by, for each driver and each race, subtracting the driver's grid position from" \
               " their finishing position then taking the mean."
    mean_spmfp_scatter.add_layout(Title(text=subtitle, text_font_style="italic"), "above")

    mean_spmfp_scatter.scatter(x="driver_final_standing", y="mean_spmfp", source=source, size=8, color="color")

    # Labels
    labels = LabelSet(x="driver_final_standing", y="mean_spmfp", text="short_name", level="glyph",
                      x_offset=0.7, y_offset=0.7, source=ColumnDataSource(data=source.to_dict(orient="list")),
                      render_mode="canvas", text_color="white", text_font_size="10pt", angle=0.2 * math.pi / 2)
    mean_spmfp_scatter.add_layout(labels)

    # Adjust axes a bit
    min_spmfp = source["mean_spmfp"].min()
    max_spmfp = source["mean_spmfp"].max()
    mean_spmfp_scatter.x_range = Range1d(0, 1.1 * source["driver_final_standing"].max())
    mean_spmfp_scatter.y_range = Range1d((1.1 if min_spmfp < 0 else 0.8) * min_spmfp,
                                         1.1 * max_spmfp)

    # Hover tooltip
    mean_spmfp_scatter.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Name", "@full_name"),
        ("Constructor", "@constructor_name"),
        ("Final Standing", "@driver_final_standing"),
        ("Mean Starting Position minus Finish Position", "@mean_spmfp")
    ]))

    # Crosshair
    mean_spmfp_scatter.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

    # Add some annotations to the graph
    label_kwargs = dict(x=1,
                        render_mode="canvas",
                        text_color="white",
                        text_font_size="12pt",
                        border_line_color="white",
                        border_line_alpha=0.7)
    label_top = Label(y=max_spmfp + 0.2, text=" Tended to finish higher than they started ", **label_kwargs)
    label_bottom = Label(y=min_spmfp - 0.2, text=" Finish lower than started ", **label_kwargs)
    mean_spmfp_scatter.add_layout(label_top)
    mean_spmfp_scatter.add_layout(label_bottom)

    # Highlight the y = 0 line
    line = Span(line_color="white", location=0, dimension="width", line_alpha=0.5, line_width=3)
    mean_spmfp_scatter.add_layout(line)

    return mean_spmfp_scatter


def generate_teams_and_drivers(year_results, year_races):
    """
    Generates a table of all of the teams and their respective drivers.
    :param year_results: Year results
    :param year_races: Year races
    :return: Table layout
    """
    # TODO scrape chassis, engine, entrant full name info
    # Simply format with all of the bodies
    table_format = """ 
        <table>
        <thead>
        <tr>
            <th scope="col">Constructor</th>
            <th scope="col">Drivers</th>
            <th scope="col">Rounds</th>
        </tr>
        </thead>
        {}
        </table>
        """

    # Format with number of sub-rows, constructor name, first driver name, first driver rounds
    tbody_format = """
        <tbody>
        <tr>
            <th rowspan="{}" scope="rowgroup" style="font-size: 16px;">{}</th>
            <th scope="row">{}</th>
            <td>{}</td>
        </tr>
        {}
        </tbody>
        """

    # Format with driver name, driver rounds
    tr_format = """
        <tr>
            <th scope="row">{}</th>
            <td>{}</td>
        </tr>
        """

    rows = []
    for constructor_id in year_results["constructorId"].unique():
        drivers = defaultdict(lambda: [])  # driver: rounds
        constructor_results = year_results[year_results["constructorId"] == constructor_id]
        for idx, row in constructor_results.iterrows():
            drivers[get_driver_name(row["driverId"])].append(year_races.loc[row["raceId"], "round"])
        # Now convert the list of rounds to a string
        drivers_list = []
        for k, v in drivers.items():
            drivers_list.append([k, rounds_to_str(v, year_races.shape[0])])
        num_drivers = str(len(drivers))
        constructor_name = get_constructor_name(constructor_id)
        trs = []
        for driver in drivers_list[1:]:
            trs.append(tr_format.format(driver[0], driver[1]))
        tbody = tbody_format.format(num_drivers, constructor_name, drivers_list[0][0], drivers_list[0][1],
                                    "\n".join(trs))
        rows.append(tbody)

    text = table_format.format("\n".join(rows))

    title = Div(text=u"<h2>Teams and Drivers \u2014 Who raced for who?</h2>")
    table = bokeh.layouts.row([Div(text=text)], sizing_mode="stretch_width")

    return column([title, table], sizing_mode="stretch_width")


def generate_races_info(year_races, year_qualifying, year_results, year_fastest_lap_data):
    """
    Generates a summary table of all of the races of the season.
    :param year_races: Year races
    :param year_qualifying: Year qualifying
    :param year_results: Year results
    :param year_fastest_lap_data: Year fastest lap data
    :return: Table layout
    """
    # Round, Date, Grand Prix Name, Pole Position, Fastest Lap, Winning Driver, Winning Constructor
    rows = pd.DataFrame(columns=["Round", "Date", "Grand Prix", "Pole Position", "Fastest Lap",
                                 "Winning Driver", "Winning Constructor"])
    have_fastest_lap = year_fastest_lap_data["fastest_lap_time_millis"].isna().sum() < year_fastest_lap_data.shape[0]

    for rid, row in year_races.sort_values(by="round").iterrows():
        round = str(row["round"])
        name = row["name"]
        date = row["datetime"]
        date = date.split(" ")[0]
        if "1990-01-01" in date:
            date = None
        else:
            date = datetime.strptime(date, "%Y-%m-%d").strftime("%d %B").lstrip("0")
        pole_position = year_qualifying[(year_qualifying["raceId"] == rid) & (year_qualifying["position"] == 1)]
        if pole_position.shape[0] > 0:
            pole_position = get_driver_name(pole_position["driverId"].values[0])
        else:
            pole_position = None
        race_results = year_results[year_results["raceId"] == rid]
        if have_fastest_lap:
            race_fastest_lap_data = year_fastest_lap_data[year_fastest_lap_data["raceId"] == rid]
            fastest_lap = race_fastest_lap_data[race_fastest_lap_data["rank"].fillna("").str.match(" 1")]
            fastest_did = fastest_lap["driver_id"].values[0]
            fastest_lap_lap = race_results[race_results["driverId"] == fastest_did]["fastestLap"].values[0]
            fastest_lap_lap = None if np.isnan(fastest_lap_lap) else int(fastest_lap_lap)
            fastest_lap_info = (fastest_lap["name"].values[0], fastest_lap["constructor_name"].values[0],
                                fastest_lap["fastest_lap_time_str"].values[0], fastest_lap_lap)
        else:
            fastest_lap_info = ()

        if len(fastest_lap_info) > 0:
            fastest_lap = fastest_lap_info[0]
        else:
            fastest_lap = None
        winner = race_results[race_results["position"] == 1]
        winning_driver = get_driver_name(winner["driverId"].values[0])
        winning_constructor = get_constructor_name(winner["constructorId"].values[0])
        rows = rows.append({
            "Round": round,
            "Date": date,
            "Grand Prix": name,
            "Pole Position": pole_position,
            "Fastest Lap": fastest_lap,
            "Winning Driver": winning_driver,
            "Winning Constructor": winning_constructor
        }, ignore_index=True)

    rows = rows.dropna(axis=1, how="all")
    rows = [rows.columns.values.tolist()] + rows.values.tolist()
    table_html = HTML_table.table(rows)

    title = Div(text=u"<h2>Grand Prix \u2014 Summary of all of the races</h2>")
    table = Div(text=table_html)
    c = [title, table]

    if year_races["year"].values[0] < 2003:
        disclaimer = Div(text="Some qualifying information may be inaccurate. Qualifying data is only fully "
                              "supported for 2003 onward.")
        c.append(disclaimer)

    return column(c, sizing_mode="stretch_width")


def generate_wdc_results(year_results, year_driver_standings, year_races):
    """
    Generates a table showing the results for every driver at every Grand Prix.
    :param year_results: Year results
    :param year_driver_standings: Year driver standings
    :param year_races: Year races
    :return: Table layout
    """
    # Pos, Driver, <column for every round>
    final_rid = year_driver_standings["raceId"].max()
    final_standings = year_driver_standings[year_driver_standings["raceId"] == final_rid].set_index("driverId")
    races = year_races.index.map(lambda x: get_race_name(x, include_country=False, line_br="<br>", use_shortened=True))
    races = races.values.tolist()

    rows = []
    for driver_id in year_results["driverId"].unique():
        if driver_id in final_standings.index:
            standings = final_standings.loc[driver_id]
            finishing_position = str(standings["position"])
            points = standings["points"]
        else:
            finishing_position = "~~"
            points = 0
        if points == "":
            points = 0
        if abs(int(points) - points) < 1e-3:
            points = int(points)
        points = str(points)
        finishing_position = finishing_position.rjust(2)
        name = get_driver_name(driver_id)
        row = [finishing_position, name]
        for rid in year_races.index:
            driver_result = year_results[(year_results["raceId"] == rid) & (year_results["driverId"] == driver_id)]
            if driver_result.shape[0] == 0:
                position = ""
            else:
                position = position_text_to_str(driver_result["positionText"].values[0])
            row.append(position)
        row.append(points)
        rows.append(row)

    columns = ["Pos.", "Driver"] + races + ["Points"]
    rows = pd.DataFrame(rows, columns=columns).sort_values(by="Pos.")
    table = Table()
    header = []
    for c in columns:
        header.append(TableCell(c, style="text-align: center;"))
    table.rows.append(TableRow(cells=header))

    def pos_to_color(pos):
        if pos == "" or pos == " " or pos == "  ":
            return RGB(60, 60, 60).to_hex(), "color:white"
        if pos == "1":
            return RGB(250, 247, 82).to_hex(), "color:black;"
        elif pos == "2":
            return RGB(207, 207, 207).to_hex(), "color:black;"
        elif pos == "3":
            return RGB(191, 141, 90).to_hex(), "color:black;"
        elif pos == "RET":
            return RGB(70, 62, 75).to_hex(), "color:white"
        elif pos == "DNQ":
            return RGB(248, 209, 208).to_hex(), "color:black"
        elif pos == "NC":
            return RGB(207, 208, 251).to_hex(), "color:black"
        elif pos == "DSQ":
            return RGB(0, 0, 0).to_hex(), "color:white"
        elif int(pos) <= 10:
            return RGB(79, 166, 71).to_hex(), "color:black"
        else:
            return RGB(69, 70, 161).to_hex(), "color:white"

    for idx, row in rows.iterrows():
        cells = []
        for pos in row.values.tolist()[2:-1]:
            color, style = pos_to_color(pos)
            colored_cell = TableCell(pos, style=f"background-color:{color};{style};")
            cells.append(colored_cell)
        tablerow = TableRow(cells=[TableCell(row["Pos."]), TableCell(row["Driver"])] + cells +
                                  [TableCell(row["Points"])])
        table.rows.append(tablerow)

    htmlcode = str(table)

    title = Div(text="""<h2 style="margin-bottom:0px;">World Driver's Championship Table</h2>""")
    subtitle = Div(text="<i>The coloring is based on finishing status, and top 10 are colored differently.</i>")
    table = bokeh.layouts.row([Div(text=htmlcode)], sizing_mode="stretch_width")

    return column([title, subtitle, table], sizing_mode="stretch_width")


def generate_dnf_table(year_results):
    """
    Generates a table showing the number of DNFs each constructor and driver has
    :param year_results: Year results
    :return: Table layout
    """
    # Simply format with all of the bodies
    # Constructor, Finished, Mechanical DNFs, Crash DNFs, Drivers, Reliability DNFs, Crash DNFs
    # For the DNF rows, have, for example, 5 / 17 (29%)
    table_format = """ 
        <table>
        <tr>
            <th>Constructor</th>
            <th>Finished</th>
            <th>Mechanical DNFs</th>
            <th>Crash DNFs</th>
            <th>Driver</th>
            <th>Finished</th>
            <th>Mechanical DNFs</th>
            <th>Crash DNFs</th>
        </tr>
        {}
        </table>
        """

    # Format with: num drivers, constructor name, num drivers, finished, num drivers, mechanical DNFs, num drivers,
    # crash DNFs, first driver name, first driver finished, first driver mechanical DNFs, first driver crash DNFs,
    # rest of the trs
    tbody_format = """
        <tr>
            <th rowspan="{}">{}</th>
            <th rowspan="{}">{}</th>
            <th rowspan="{}">{}</th>
            <th rowspan="{}">{}</th>
            <th>{}</th>
            <td>{}</td>
            <td>{}</td>
            <td>{}</td>
        </tr>
        {}
        """

    # Format with driver name, driver mechanical DNFs, driver crash DNFs
    tr_format = """
        <tr>
            <th>{}</th>
            <td>{}</td>
            <td>{}</td>
            <td>{}</td>
        </tr>
        """

    def format_dnfs(finished, crashed, mechanical):
        total = finished + crashed + mechanical
        if total <= 0.01:
            return "0 / 0 (0%)", "0 / 0 (0%)", "0 / 0 (0%)"
        finished_str = f"{finished} / {total} ({round(100 * finished / total, 2)}%)"
        crashed_str = f"{crashed} / {total} ({round(100 * crashed / total, 2)}%)"
        mechanical_str = f"{mechanical} / {total} ({round(100 * mechanical / total, 2)}%)"
        return finished_str, mechanical_str, crashed_str

    rows = []
    for constructor_id in year_results["constructorId"].unique():
        constructor_results = year_results[year_results["constructorId"] == constructor_id]
        constructor_name = get_constructor_name(constructor_id)
        constructor_num_finished = 0
        constructor_num_crash_dnfs = 0
        constructor_num_mechanical_dnfs = 0
        # name, finished, crash, mechanical
        drivers = []
        for driver_id in constructor_results["driverId"].unique():
            driver_results = constructor_results[constructor_results["driverId"] == driver_id]
            statuses = driver_results["statusId"]
            classifications = statuses.apply(get_status_classification)
            name = get_driver_name(driver_id)
            driver_finished = classifications[classifications == "finished"].shape[0]
            driver_num_crash_dnq = classifications[classifications == "crash"].shape[0]
            driver_num_mechanical_dnfs = classifications[classifications == "mechanical"].shape[0]
            constructor_num_finished += driver_finished
            constructor_num_crash_dnfs += driver_num_crash_dnq
            constructor_num_mechanical_dnfs += driver_num_mechanical_dnfs
            formatted = format_dnfs(driver_finished, driver_num_crash_dnq, driver_num_mechanical_dnfs)
            drivers.append([name, formatted[0], formatted[1], formatted[2]])

        trs = []
        for driver_info in drivers[1:]:
            trs.append(tr_format.format(driver_info[0], driver_info[1], driver_info[2], driver_info[3]))
        num_drivers = len(drivers)
        formatted = format_dnfs(constructor_num_finished, constructor_num_mechanical_dnfs,
                                constructor_num_mechanical_dnfs)
        args = [
            num_drivers, constructor_name,
            num_drivers, formatted[0],
            num_drivers, formatted[1],
            num_drivers, formatted[2],
            drivers[0][0], drivers[0][1], drivers[0][2], drivers[0][3],
            "\n".join(trs)
        ]
        tbody = tbody_format.format(*args)
        rows.append(tbody)

    text = table_format.format("\n".join(rows))

    title = Div(text=u"""<h2 style="margin-bottom:0px;">DNF Chart \u2014 
    How Often Did every Driver and Constructor Retire</h2>""")
    subtitle = Div(text="<i>Finished shows number of started and finished races, crashes include self-enforced errors,"
                        " the denominator of the fraction is races entered.</i>")
    table = Div(text=text)

    return column([title, subtitle, table], sizing_mode="stretch_width")


def generate_error_layout():
    """
    Creates an error layout in the case where the year is not valid.
    :return: The error layout
    """
    text = "Somehow, you have selected an invalid season. The seasons we have data on are..."
    text += "<ul>"
    for year in seasons["year"].unique():
        text += f"<li>{str(year)}</li>"
    text += "</ul><br>"
    return Div(text=text)
