import itertools
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
    CrosshairTool, LabelSet, Label, Span, TableColumn, DataTable, LinearAxis, NumeralTickFormatter
from bokeh.palettes import Category20_20
from bokeh.plotting import figure
from bokeh.transform import cumsum
from html_table import HTML_table
from html_table.HTML_table import TableCell, Table, TableRow
from data_loading.data_loader import load_seasons, load_driver_standings, load_races, load_results, \
    load_constructor_standings, load_lap_times, load_qualifying, load_fastest_lap_data
from mode import driver
from utils import PLOT_BACKGROUND_COLOR, get_line_thickness, get_driver_name, get_constructor_name, \
    ColorDashGenerator, get_race_name, position_text_to_str, get_status_classification, rounds_to_str, int_to_ordinal, \
    vdivider

seasons = load_seasons()
driver_standings = load_driver_standings()
constructor_standings = load_constructor_standings()
races = load_races()
results = load_results()
lap_times = load_lap_times()
qualifying = load_qualifying()
fastest_lap_data = load_fastest_lap_data()

# TODO add stats layout (see Trello)


def get_layout(year_id=-1, **kwargs):
    if year_id not in seasons.index.unique():
        return generate_error_layout()

    # Generate some useful slices
    year_races = races[races["year"] == year_id]
    year_driver_standings = driver_standings[driver_standings["raceId"].isin(year_races.index)].sort_values(by="raceId")
    year_constructor_standings = constructor_standings[constructor_standings["raceId"].isin(year_races.index)]\
        .sort_values(by="raceId")
    year_results = results[results["raceId"].isin(year_races.index)]
    year_qualifying = qualifying[qualifying["raceId"].isin(year_races.index)]
    year_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"].isin(year_races.index)]

    logging.info(f"Generating layout for mode YEAR in year, year_id={year_id}")

    header = Div(text=f"<h2>What did the {year_id} season look like?</h2>")

    # Generate WDC plot
    wdc_plot = generate_wdc_plot(year_driver_standings, year_results)

    # Generate constructor's plot
    constructors_plot = generate_wcc_plot(year_constructor_standings, year_results)

    # Generate position vs mean lap time rank plot
    position_mltr_scatter = generate_mltr_position_scatter(year_fastest_lap_data, year_results,
                                                           year_driver_standings, year_constructor_standings)

    # Generate mean finish start position vs WDC finish position scatter plot
    msp_position_scatter = generate_msp_position_scatter(year_results, year_driver_standings)

    # Start pos vs finish pos scatter plot
    spvpfp_scatter = generate_spvfp_scatter(year_results, year_races, year_driver_standings)

    # WCC results table
    wcc_results_table = generate_wcc_results_table(year_results, year_races, year_constructor_standings)

    # Wins pie chart
    wins_pie_chart = generate_wins_pie_plots(year_results)

    # Generate the teams and drivers table
    teams_and_drivers = generate_teams_and_drivers_table(year_results, year_races)

    # Generate races info
    races_info = generate_races_info_table(year_races, year_qualifying, year_results, year_fastest_lap_data)

    # Generate WDC table
    wdc_table, driver_win_source, constructor_win_source = generate_wdc_results_table(year_results,
                                                                                      year_driver_standings, year_races)

    # Win plots
    win_plots = generate_win_plots(driver_win_source, constructor_win_source)

    # Generate DNF table
    dnf_table = generate_dnf_table(year_results)

    # Bring it all together
    middle_spacer = Spacer(width=5, background=PLOT_BACKGROUND_COLOR)
    layout = column([header,
                     wdc_plot, middle_spacer,
                     constructors_plot, middle_spacer,
                     wins_pie_chart, middle_spacer,
                     row([position_mltr_scatter, msp_position_scatter], sizing_mode="stretch_width"), middle_spacer,
                     win_plots, middle_spacer,
                     row([spvpfp_scatter, vdivider(), wcc_results_table], sizing_mode="stretch_width"), middle_spacer,
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
        driver_standings["position_str"] = driver_standings["position"].apply(int_to_ordinal)
        driver_standings["final_position_str"] = driver_standings["final_position"].apply(int_to_ordinal)

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
        ("Current Position", "@position_str"),
        ("Final Position", "@final_position_str"),
        ("Constructor", "@constructor_name")
    ]))

    # Crosshair
    wdc_plot.add_tools(CrosshairTool(dimensions="both", line_color="white", line_alpha=0.6))

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
        constructor_standings = year_constructor_standings[
            year_constructor_standings["constructorId"] == constructor_id]
        constructor_standings = constructor_standings.copy()
        name = get_constructor_name(constructor_standings["constructorId"].values[0])
        constructor_standings["name"] = name

        final_standing = final_standings[final_standings["constructorId"] == constructor_id]["position"].values[0]
        constructor_standings["final_position"] = final_standing
        constructor_standings["position_str"] = constructor_standings["position"].apply(int_to_ordinal)
        constructor_standings["final_position_str"] = constructor_standings["final_position"].apply(int_to_ordinal)

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
        ("Current Position", "@position_str"),
        ("Final Position", "@final_position_str")
    ]))

    # Crosshair
    constructors_plot.add_tools(CrosshairTool(dimensions="both", line_color="white", line_alpha=0.6))

    return constructors_plot


def generate_mltr_position_scatter(year_fastest_lap_data, year_results, year_driver_standings,
                                   year_constructor_standings):
    """
    Driver finish position (WDC) vs their constructor's mean lap time rank to get a sense of who did well despite a
    worse car. Basically MLTR vs FP plot but for a whole season.
    :param year_fastest_lap_data: Year fastest lap data
    :param year_results: Year results
    :param year_driver_standings: Year driver standings
    :param year_constructor_standings: Year constructor standings
    :return: Position vs mean lap time plot
    """
    # TODO change to mean lap time percent
    logging.info("Generating position vs mean lap time rank scatter")
    if year_fastest_lap_data.shape[0] == 0:
        return Div()

    final_rid = year_results["raceId"].max()
    final_driver_standings = year_driver_standings[year_driver_standings["raceId"] == final_rid]
    final_driver_standings = final_driver_standings.set_index("driverId")
    num_drivers = final_driver_standings.shape[0]
    final_constructor_standings = year_constructor_standings[year_constructor_standings["raceId"] == final_rid]
    final_constructor_standings = final_constructor_standings.set_index("constructorId")

    source = pd.DataFrame(columns=["short_name", "full_name", "constructor_name",
                                   "driver_final_standing", "driver_final_standing_str",
                                   "constructor_final_standing", "constructor_final_standing_str",
                                   "constructor_mean_rank", "color"])
    position_mlt_scatter = figure(
        title=u"Constructor Mean Lap Time Rank versus WDC Position \u2014 Who did well with a poor car?",
        x_axis_label="Constructor Mean Lap Time Rank",
        y_axis_label="World Driver's Championship Final Position",
        tools="pan,reset,save",
        x_range=Range1d(0, 12, bounds=(0, 14)),
        y_range=Range1d(0, 22, bounds=(0, 60)),
        plot_height=30 * min(num_drivers, 30)
    )
    position_mlt_scatter.xaxis.ticker = FixedTicker(ticks=np.arange(2, 100, 2).tolist() + [1])
    position_mlt_scatter.yaxis.ticker = FixedTicker(ticks=np.arange(4, 100, 4).tolist() + [1])

    explanation = "The x axis is computed by finding the average lap time of every constructor at every race, and " \
                  "then for every race, ranking each constructor based on average lap time. Those ranks are then " \
                  "averaged. If a driver switches constructors during the season, the constructor who they were with " \
                  "the longest is used."

    color_gen = ColorDashGenerator()
    constructor_avg_lap_ranks = year_fastest_lap_data.groupby("constructor_id").agg("mean")["avg_lap_time_rank"] / 2
    for driver_id in year_results["driverId"].unique():
        driver_results = year_results[year_results["driverId"] == driver_id]
        full_name = get_driver_name(driver_id)
        short_name = get_driver_name(driver_id, include_flag=False, just_last=True)
        cid = driver_results["constructorId"].mode().values[0]
        constructor_name = get_constructor_name(cid)
        if cid in constructor_avg_lap_ranks.index:
            constructor_mean_rank = constructor_avg_lap_ranks.loc[cid]
        else:
            constructor_mean_rank = np.nan
        if driver_id in final_driver_standings.index:
            driver_final_standing = final_driver_standings.loc[driver_id, "position"]
        elif driver_id in year_driver_standings["driverId"]:
            driver_final_standing = year_driver_standings[year_driver_standings["driverId"] == driver_id]["position"]
            driver_final_standing = driver_final_standing[-1]
        else:
            continue
        if cid in final_constructor_standings.index:
            constructor_final_standing = final_constructor_standings.loc[cid, "position"]
        else:
            constructor_final_standing = np.nan

        color, _ = color_gen.get_color_dash(driver_id, cid)

        source = source.append({
            "short_name": short_name,
            "full_name": full_name,
            "constructor_name": constructor_name,
            "driver_final_standing": driver_final_standing,
            "driver_final_standing_str": int_to_ordinal(driver_final_standing),
            "constructor_final_standing": constructor_final_standing,
            "constructor_final_standing_str": int_to_ordinal(constructor_final_standing),
            "constructor_mean_rank": constructor_mean_rank,
            "color": color
        }, ignore_index=True)

    position_mlt_scatter.scatter(x="constructor_mean_rank", y="driver_final_standing", source=source, size=8,
                                 color="color")
    position_mlt_scatter.line(x=[-60, 60], y=[-120, 120], color="white", line_alpha=0.5)

    # Labels
    labels = LabelSet(x="constructor_mean_rank", y="driver_final_standing", text="short_name", level="glyph",
                      x_offset=0.7, y_offset=0.7, source=ColumnDataSource(data=source.to_dict(orient="list")),
                      render_mode="canvas", text_color="white", text_font_size="10pt")
    position_mlt_scatter.add_layout(labels)

    text_label_kwargs = dict(render_mode="canvas",
                             text_color="white",
                             text_font_size="12pt",
                             border_line_color="white",
                             border_line_alpha=0.7)
    label1 = Label(x=1, y=21, text=" Finish lower than expected ", **text_label_kwargs)
    label2 = Label(x=7.5, y=0.25, text=" Finish higher than expected ", **text_label_kwargs)
    position_mlt_scatter.add_layout(label1)
    position_mlt_scatter.add_layout(label2)

    # Hover tooltip
    position_mlt_scatter.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Name", "@full_name"),
        ("Constructor", "@constructor_name"),
        ("Final Standing", "@driver_final_standing_str"),
        ("Constructor Final Standing", "@constructor_final_standing_str"),
        ("Constructor Mean Lap Time Rank", "@constructor_mean_rank")
    ]))

    # Crosshair
    position_mlt_scatter.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

    return column([position_mlt_scatter, Div(text=explanation)], sizing_mode="stretch_width")


def generate_wins_pie_plots(year_results):
    """
    Generates 2 pie charts for winners this year, 1 for drivers and 1 for constructors.
    :param year_results: Year results
    :return: Wins plot layout
    """
    logging.info("Generating wins pie plot")
    year_wins = year_results[year_results["position"] == 1]
    wins_source = year_wins.groupby("driverId").agg("count").rename(columns={"raceId": "num_wins"})["num_wins"]
    wins_source = pd.DataFrame(wins_source)
    wins_source["pct_wins"] = wins_source["num_wins"] / wins_source["num_wins"].sum()
    wins_source["pct_wins_str"] = wins_source["pct_wins"].apply(lambda x: str(100 * round(x, 1)) + "%")
    wins_source["angle"] = 2 * math.pi * wins_source["num_wins"] / wins_source["num_wins"].sum()
    wins_source["constructorId"] = None
    wins_source["color"] = None
    wins_source["driver_name"] = None
    wins_source["constructor_name"] = None
    wins_source["alpha"] = None
    gen = ColorDashGenerator(dashes=[1, 0.5])
    for idx, source_row in wins_source.iterrows():
        did = idx
        cid = year_results[year_results["driverId"] == did]["constructorId"].mode()
        if cid.shape[0] > 0:
            cid = cid.values[0]
        else:
            cid = 0
        wins_source.loc[did, "constructorId"] = cid
        color, alpha = gen.get_color_dash(did, cid)
        wins_source.loc[did, "color"] = color
        wins_source.loc[did, "alpha"] = alpha
        wins_source.loc[did, "driver_name"] = get_driver_name(did)
        wins_source.loc[did, "constructor_name"] = get_constructor_name(cid)

    driver_pie_chart = figure(title="Pie Chart", toolbar_location=None,
                              tools="hover", tooltips="Name: @driver_name<br>"
                                                      "Wins: @num_wins (@pct_wins_str)<br>"
                                                      "Constructor: @constructor_name",
                              x_range=(-0.5, 0.5), y_range=(-0.5, 0.5))

    driver_pie_chart.wedge(x=0, y=0, radius=0.4, start_angle=cumsum("angle", include_zero=True),
                           end_angle=cumsum("angle"), line_color="white", legend_field="driver_name", color="color",
                           source=wins_source, fill_alpha="alpha")
    driver_pie_chart.axis.axis_label = None
    driver_pie_chart.axis.visible = False
    driver_pie_chart.grid.grid_line_color = None

    wins_source = year_wins.groupby("constructorId").agg("count").rename(columns={"raceId": "num_wins"})["num_wins"]
    wins_source = pd.DataFrame(wins_source)
    wins_source["angle"] = 2 * math.pi * wins_source["num_wins"] / wins_source["num_wins"].sum()
    wins_source["pct_wins"] = wins_source["num_wins"] / wins_source["num_wins"].sum()
    wins_source["pct_wins_str"] = wins_source["pct_wins"].apply(lambda x: str(100 * round(x, 1)) + "%")
    wins_source["color"] = None
    wins_source["constructor_name"] = None
    gen = ColorDashGenerator()
    for idx, source_row in wins_source.iterrows():
        cid = idx
        color, _ = gen.get_color_dash(None, cid)
        wins_source.loc[cid, "color"] = color
        wins_source.loc[cid, "constructor_name"] = get_constructor_name(cid)

    constructor_pie_chart = figure(title="Pie Chart", toolbar_location=None, tools="hover",
                                   tooltips="Name: @constructor_name<br>"
                                            "Wins: @num_wins (@pct_wins_str)<br>",
                                   x_range=(-0.5, 0.5), y_range=(-0.5, 0.5))

    constructor_pie_chart.wedge(x=0, y=0, radius=0.4, start_angle=cumsum("angle", include_zero=True),
                                end_angle=cumsum("angle"), line_color="white", legend_field="constructor_name",
                                color="color", source=wins_source)
    constructor_pie_chart.axis.axis_label = None
    constructor_pie_chart.axis.visible = False
    constructor_pie_chart.grid.grid_line_color = None

    return row([driver_pie_chart, constructor_pie_chart], sizing_mode="stretch_width")


def generate_msp_position_scatter(year_results, year_driver_standings):
    """
    Mean start position vs WDC finish pos scatter
    :param year_results: Year results
    :param year_driver_standings: Year driver standings
    :return: MSP vs position scatter layout
    """
    logging.info("Generating mean SP vs position scatter plot")
    final_rid = year_results["raceId"].max()
    final_driver_standings = year_driver_standings[year_driver_standings["raceId"] == final_rid]
    final_driver_standings = final_driver_standings.set_index("driverId")
    num_drivers = final_driver_standings.shape[0]

    source = pd.DataFrame(columns=["short_name", "full_name", "constructor_name",
                                   "driver_final_standing", "driver_final_standing_str",
                                   "mean_sp",
                                   "color"])

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

        source = source.append({
            "short_name": short_name,
            "full_name": full_name,
            "constructor_name": constructor_name,
            "driver_final_standing": final_standing,
            "driver_final_standing_str": int_to_ordinal(final_standing),
            "mean_sp": start_position.mean(),
            "color": color
        }, ignore_index=True)

    mean_sp_scatter = figure(
        title="Mean Start Position vs WDC Finish Position",
        x_axis_label="Mean Start Position",
        y_axis_label="WDC Final Position",
        x_range=Range1d(0, 22, bounds=(0, 60)),
        y_range=Range1d(0, 22, bounds=(0, 60)),
        tools="pan,reset,save",
        plot_height=30 * min(num_drivers, 30)
    )
    mean_sp_scatter.line(x=[-60, 60], y=[-60, 60], color="white", line_alpha=0.5)
    mean_sp_scatter.xaxis.ticker = FixedTicker(ticks=np.arange(5, 100, 5).tolist() + [1])
    mean_sp_scatter.yaxis.ticker = FixedTicker(ticks=np.arange(5, 100, 5).tolist() + [1])

    mean_sp_scatter.scatter(x="mean_sp", y="driver_final_standing", source=source, size=8, color="color")

    # Labels
    labels = LabelSet(x="mean_sp", y="driver_final_standing", text="short_name", level="glyph",
                      x_offset=0.7, y_offset=0.7, source=ColumnDataSource(data=source.to_dict(orient="list")),
                      render_mode="canvas", text_color="white", text_font_size="10pt")
    mean_sp_scatter.add_layout(labels)

    # Hover tooltip
    mean_sp_scatter.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Name", "@full_name"),
        ("Constructor", "@constructor_name"),
        ("Final Standing", "@driver_final_standing_str"),
        ("Mean Starting Position", "@mean_sp")
    ]))

    # Crosshair
    mean_sp_scatter.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

    # Add some annotations to the graph
    text_label_kwargs = dict(render_mode="canvas",
                             text_color="white",
                             text_font_size="12pt",
                             border_line_color="white",
                             border_line_alpha=0.7)
    label1 = Label(x=1, y=21, text=" Finish lower than expected ", **text_label_kwargs)
    label2 = Label(x=7.5, y=0.25, text=" Finish higher than expected ", **text_label_kwargs)
    mean_sp_scatter.add_layout(label1)
    mean_sp_scatter.add_layout(label2)

    # Highlight the y = 0 line
    line = Span(line_color="white", location=0, dimension="width", line_alpha=0.5, line_width=3)
    mean_sp_scatter.add_layout(line)

    return mean_sp_scatter


def generate_spvfp_scatter(year_results, year_races, year_driver_standings):
    """
    Start position vs finish position scatter
    :param year_results: Year results
    :param year_races: Year races
    :param year_driver_standings: Year driver races
    :return: Start pos vs finish pos scatter layout
    """
    return driver.generate_spvfp_scatter(year_results, year_races, year_driver_standings, color_drivers=True)


def generate_win_plots(driver_win_source, constructor_win_source):
    logging.info("Generating win plots")
    # Yes, I partially re-wrote this method for efficiency

    # Drivers
    driver_win_source["win_pct"] = driver_win_source["wins"] / driver_win_source["num_races"]
    driver_win_source["podium_pct"] = driver_win_source["podiums"] / driver_win_source["num_races"]
    driver_win_source["dnf_pct"] = driver_win_source["dnfs"] / driver_win_source["num_races"]
    driver_win_source["win_pct_str"] = driver_win_source["win_pct"].apply(lambda x: str(round(100 * x, 1)) + "%")
    driver_win_source["podium_pct_str"] = driver_win_source["podium_pct"].apply(lambda x: str(round(100 * x, 1)) + "%")
    driver_win_source["dnf_pct_str"] = driver_win_source["dnf_pct"].apply(lambda x: str(round(100 * x, 1)) + "%")

    max_podium = driver_win_source["podiums"].max()
    max_dnfs = driver_win_source["dnfs"].max()
    max_x = driver_win_source["roundNum"].max()
    driver_win_plot = figure(
        title=u"Driver Win Plot \u2014 Wins and Win Percent",
        y_axis_label="",
        x_axis_label="Year",
        x_range=Range1d(1, max_x, bounds=(1, max_x + 3)),
        tools="pan,xbox_zoom,reset,box_zoom,wheel_zoom,save",
        y_range=Range1d(0, max(max_podium, max_dnfs), bounds=(0, 1000))
    )
    driver_podium_plot = figure(
        title=u"Driver Podium Plot \u2014 Podiums and Podium Percent",
        y_axis_label="",
        x_axis_label="Year",
        x_range=Range1d(1, max_x, bounds=(1, max_x + 3)),
        tools="pan,xbox_zoom,reset,box_zoom,wheel_zoom,save",
        y_range=Range1d(0, max(max_podium, max_dnfs), bounds=(0, 1000))
    )
    driver_dnf_plot = figure(
        title=u"Driver DNF Plot \u2014 DNFs and DNF Percent",
        y_axis_label="",
        x_axis_label="Year",
        x_range=Range1d(1, max_x, bounds=(1, max_x + 3)),
        tools="pan,xbox_zoom,reset,box_zoom,wheel_zoom,save",
        y_range=Range1d(0, max(max_podium, max_dnfs), bounds=(0, 1000))
    )
    plots = [driver_win_plot, driver_podium_plot, driver_dnf_plot]

    max_dnf_pct = driver_win_source["dnf_pct"].max()
    if max_podium == 0:
        return Div()
    if max_podium > max_dnfs:
        k = max_podium / 1
    elif max_dnf_pct > 0:
        k = max_dnfs / 1
    else:
        k = 1
    driver_win_source["podium_pct_scaled"] = k * driver_win_source["podium_pct"]
    driver_win_source["win_pct_scaled"] = k * driver_win_source["win_pct"]
    driver_win_source["dnf_pct_scaled"] = k * driver_win_source["dnf_pct"]

    kwargs = {
        "x": "roundNum",
        "line_width": 2,
        "muted_alpha": 0.02
    }
    color_gen = itertools.cycle(Category20_20)
    legends = [[], [], []]
    for driver_id in driver_win_source["driver_id"].unique():
        name = get_driver_name(driver_id, include_flag=False, just_last=True)
        win_source = driver_win_source[driver_win_source["driver_id"] == driver_id]
        kwargs["source"] = win_source
        color = color_gen.__next__()
        wins_line = driver_win_plot.line(y="wins", color=color, line_alpha=0.6, **kwargs)
        win_pct_line = driver_win_plot.line(y="win_pct_scaled", color=color, line_dash="dashed", **kwargs)
        podiums_line = driver_podium_plot.line(y="podiums", color=color, line_alpha=0.6, **kwargs)
        podium_pct_line = driver_podium_plot.line(y="podium_pct_scaled", color=color, line_dash="dashed", **kwargs)
        dnfs_line = driver_dnf_plot.line(y="dnfs", color=color, line_alpha=0.6, **kwargs)
        dnf_pct_line = driver_dnf_plot.line(y="dnf_pct_scaled", color=color, line_dash="dashed", **kwargs)

        legends[0].append(LegendItem(label=name, renderers=[wins_line, win_pct_line]))
        legends[1].append(LegendItem(label=name, renderers=[podiums_line, podium_pct_line]))
        legends[2].append(LegendItem(label=name, renderers=[dnfs_line, dnf_pct_line]))

    for p, legend in zip(plots, legends):
        legend_layout = Legend(items=legend, location="top_right", glyph_height=15, spacing=2, inactive_fill_color="gray")
        p.add_layout(legend_layout, "right")
        p.legend.click_policy = "mute"
        p.legend.label_text_font_size = "12pt"

        # Hover tooltip
        tooltips = [
            ("Name", "@name"),
            ("Number of Races", "@num_races"),
            ("Number of Wins", "@wins (@win_pct_str)"),
            ("Number of Podiums", "@podiums (@podium_pct_str)"),
            ("Number of DNFs", "@dnfs (@dnf_pct_str)"),
        ]
        p.add_tools(HoverTool(show_arrow=False, tooltips=tooltips))

        # Crosshair tooltip
        p.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

        # Override x axis
        p.xaxis.ticker = FixedTicker(ticks=sorted(driver_win_source["roundNum"].unique()))
        p.xaxis.major_label_overrides = {src_row["roundNum"]: src_row["roundName"]
                                         for idx, src_row in driver_win_source.iterrows()}
        p.xaxis.major_label_orientation = 0.8 * math.pi / 2
        p.xaxis.major_tick_line_alpha = 0.2

    # Constructors
    constructor_win_source["win_pct"] = constructor_win_source["wins"] / constructor_win_source["num_races"]
    constructor_win_source["podium_pct"] = constructor_win_source["podiums"] / constructor_win_source["num_races"]
    constructor_win_source["dnf_pct"] = constructor_win_source["dnfs"] / constructor_win_source["num_races"]
    constructor_win_source["win_pct_str"] = constructor_win_source["win_pct"]\
        .apply(lambda x: str(round(100 * x, 1)) + "%")
    constructor_win_source["podium_pct_str"] = constructor_win_source["podium_pct"]\
        .apply(lambda x: str(round(100 * x, 1)) + "%")
    constructor_win_source["dnf_pct_str"] = constructor_win_source["dnf_pct"]\
        .apply(lambda x: str(round(100 * x, 1)) + "%")

    max_podium = constructor_win_source["podiums"].max()
    max_dnfs = constructor_win_source["dnfs"].max()
    max_x = constructor_win_source["roundNum"].max()
    constructor_win_plot = figure(
        title=u"Constructor Win Plot \u2014 Wins and Win Percent",
        y_axis_label="",
        x_axis_label="Year",
        x_range=Range1d(1, max_x, bounds=(1, max_x + 3)),
        tools="pan,xbox_zoom,reset,box_zoom,wheel_zoom,save",
        y_range=Range1d(0, max(max_podium, max_dnfs), bounds=(0, 1000))
    )
    constructor_podium_plot = figure(
        title=u"Driver Podium Plot \u2014 Podiums and Podium Percent (may exceed 100%)",
        y_axis_label="",
        x_axis_label="Year",
        x_range=Range1d(1, max_x, bounds=(1, max_x + 3)),
        tools="pan,xbox_zoom,reset,box_zoom,wheel_zoom,save",
        y_range=Range1d(0, max(max_podium, max_dnfs), bounds=(0, 1000))
    )
    constructor_dnf_plot = figure(
        title=u"Constructor DNF Plot \u2014 DNFs and DNF Percent",
        y_axis_label="",
        x_axis_label="Year",
        x_range=Range1d(1, max_x, bounds=(1, max_x + 3)),
        tools="pan,xbox_zoom,reset,box_zoom,wheel_zoom,save",
        y_range=Range1d(0, max(max_podium, max_dnfs), bounds=(0, 1000))
    )
    plots = [constructor_win_plot, constructor_podium_plot, constructor_dnf_plot]

    max_dnf_pct = constructor_win_source["dnf_pct"].max()
    if max_podium == 0:
        return Div()
    if max_podium > max_dnfs:
        k = max_podium / 1
    elif max_dnf_pct > 0:
        k = max_dnfs / 1
    else:
        k = 1
    constructor_win_source["podium_pct_scaled"] = k * constructor_win_source["podium_pct"]
    constructor_win_source["win_pct_scaled"] = k * constructor_win_source["win_pct"]
    constructor_win_source["dnf_pct_scaled"] = k * constructor_win_source["dnf_pct"]

    # Other y axis
    for p in plots:
        y_range = Range1d(start=0, end=1, bounds=(-0.02, 1000))
        p.extra_y_ranges = {"percent_range": y_range}
        axis = LinearAxis(y_range_name="percent_range")
        axis.formatter = NumeralTickFormatter(format="0.0%")
        p.add_layout(axis, "right")

    kwargs = {
        "x": "roundNum",
        "line_width": 2,
        "muted_alpha": 0.02
    }
    color_gen = itertools.cycle(Category20_20)
    legends = [[], [], []]
    for constructor_id in constructor_win_source["constructor_id"].unique():
        name = get_constructor_name(constructor_id, include_flag=False)
        win_source = constructor_win_source[constructor_win_source["constructor_id"] == constructor_id]
        kwargs["source"] = win_source
        color = color_gen.__next__()
        wins_line = constructor_win_plot.line(y="wins", color=color, line_alpha=0.6, **kwargs)
        win_pct_line = constructor_win_plot.line(y="win_pct_scaled", color=color, line_dash="dashed", **kwargs)
        podiums_line = constructor_podium_plot.line(y="podiums", color=color, line_alpha=0.6, **kwargs)
        podium_pct_line = constructor_podium_plot.line(y="podium_pct_scaled", color=color, line_dash="dashed", **kwargs)
        dnfs_line = constructor_dnf_plot.line(y="dnfs", color=color, line_alpha=0.6, **kwargs)
        dnf_pct_line = constructor_dnf_plot.line(y="dnf_pct_scaled", color=color, line_dash="dashed", **kwargs)

        legends[0].append(LegendItem(label=name, renderers=[wins_line, win_pct_line]))
        legends[1].append(LegendItem(label=name, renderers=[podiums_line, podium_pct_line]))
        legends[2].append(LegendItem(label=name, renderers=[dnfs_line, dnf_pct_line]))

    for p, legend in zip(plots, legends):
        legend_layout = Legend(items=legend, location="top_right", glyph_height=15, spacing=2,
                               inactive_fill_color="gray")
        p.add_layout(legend_layout, "right")
        p.legend.click_policy = "mute"
        p.legend.label_text_font_size = "12pt"

        # Hover tooltip
        tooltips = [
            ("Name", "@name"),
            ("Number of Races", "@num_races"),
            ("Number of Wins", "@wins (@win_pct_str)"),
            ("Number of Podiums", "@podiums (@podium_pct_str)"),
            ("Number of DNFs", "@dnfs (@dnf_pct_str)"),
        ]
        p.add_tools(HoverTool(show_arrow=False, tooltips=tooltips))

        # Crosshair tooltip
        p.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

        # Override x axis
        p.xaxis.ticker = FixedTicker(ticks=sorted(constructor_win_source["roundNum"].unique()))
        p.xaxis.major_label_overrides = {src_row["roundNum"]: src_row["roundName"]
                                         for idx, src_row in constructor_win_source.iterrows()}
        p.xaxis.major_label_orientation = 0.8 * math.pi / 2
        p.xaxis.major_tick_line_alpha = 0.2

    return column([row([driver_win_plot, constructor_win_plot], sizing_mode="stretch_width"),
                   row([driver_podium_plot, constructor_podium_plot], sizing_mode="stretch_width"),
                   row([driver_dnf_plot, constructor_dnf_plot], sizing_mode="stretch_width")],
                  sizing_mode="stretch_width")


def generate_teams_and_drivers_table(year_results, year_races):
    """
    Generates a table of all of the teams and their respective drivers.
    :param year_results: Year results
    :param year_races: Year races
    :return: Table layout
    """
    # TODO scrape chassis, engine, entrant full name info
    logging.info("Generating teams and drivers")
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


def generate_races_info_table(year_races, year_qualifying, year_results, year_fastest_lap_data):
    """
    Generates a summary table of all of the races of the season.
    :param year_races: Year races
    :param year_qualifying: Year qualifying
    :param year_results: Year results
    :param year_fastest_lap_data: Year fastest lap data
    :return: Table layout
    """
    logging.info("Generating races info")
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


def generate_wdc_results_table(year_results, year_driver_standings, year_races):
    """
    Generates a table showing the results for every driver at every Grand Prix.
    :param year_results: Year results
    :param year_driver_standings: Year driver standings
    :param year_races: Year races
    :return: Table layout, driver win source, constructor win source
    """
    logging.info("Generating WDC results")
    # Pos, Driver, <column for every round>
    final_rid = year_driver_standings["raceId"].max()
    final_standings = year_driver_standings[year_driver_standings["raceId"] == final_rid].set_index("driverId")
    races = year_races.index.map(lambda x: get_race_name(x, include_country=False, line_br="<br>", use_shortened=True))
    races = races.values.tolist()
    driver_source = pd.DataFrame(columns=["driver_id", "name", "race_id", "num_races", "wins", "podiums", "dnfs",
                                          "roundNum", "roundName"])
    constructor_source = pd.DataFrame(columns=["constructor_id", "name", "race_id", "num_races", "wins", "podiums",
                                               "dnfs", "roundNum", "roundName"])

    rows = []
    constructor_dict = {}  # constructorId: [num_races, num_wins, num_podiums, num_dnfs]
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
        num_races = 0
        num_dnfs = 0
        num_wins = 0
        num_podiums = 0
        for rid, races_row in year_races.iterrows():
            driver_result = year_results[(year_results["raceId"] == rid) & (year_results["driverId"] == driver_id)]
            if driver_result.shape[0] == 0:
                position_str = ""
                constructor_id = -1
                win = 0
                podium = 0
                dnf = 0
            else:
                position_text = driver_result["positionText"].values[0]
                constructor_id = driver_result["constructorId"].values[0]
                position_str = position_text_to_str(position_text)
                position = driver_result["position"].values[0]
                win = int(position == 1)
                podium = int(position <= 3)
                dnf = int(position_str == "RET")
            num_wins += win
            num_podiums += podium
            num_dnfs += dnf
            row.append(position_str)
            num_races += 1
            race_name = get_race_name(rid)
            driver_source = driver_source.append({
                "driver_id": driver_id,
                "name": name,
                "race_id": rid,
                "num_races": num_races,
                "wins": num_wins,
                "podiums": num_podiums,
                "dnfs": num_dnfs,
                "roundNum": races_row["round"],
                "roundName": race_name
            }, ignore_index=True)
            if constructor_id > 0:
                if constructor_id in constructor_dict:
                    curr = constructor_dict[constructor_id]
                    constructor_dict[constructor_id] = [curr[0] + 1,
                                                        curr[1] + win,
                                                        curr[2] + podium,
                                                        curr[3] + num_dnfs]
                else:
                    constructor_dict[constructor_id] = [1, win, num_podiums, num_dnfs]
                curr = constructor_dict[constructor_id]
                constructor_name = get_constructor_name(constructor_id)
                curr_slice = constructor_source[(constructor_source["name"] == constructor_name) &
                                                (constructor_source["race_id"] == rid)]
                if curr_slice.shape[0] > 0:
                    curr_iloc = curr_slice.iloc[0]
                    constructor_source.loc[curr_slice.index.values[0]] = [
                        constructor_id,
                        constructor_name,
                        rid,
                        curr_iloc["num_races"] + 1,
                        curr_iloc["wins"] + win,
                        curr_iloc["podiums"] + podium,
                        curr_iloc["dnfs"] + dnf,
                        curr_iloc["roundNum"],
                        curr_iloc["roundName"],
                    ]
                else:
                    constructor_source = constructor_source.append({
                        "constructor_id": constructor_id,
                        "name": constructor_name,
                        "race_id": rid,
                        "num_races": curr[0],
                        "wins": curr[1],
                        "podiums": curr[2],
                        "dnfs": curr[3],
                        "roundNum": races_row["round"],
                        "roundName": race_name
                    }, ignore_index=True)
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

    title = Div(text=u"""<h2 style="margin-bottom:0px;">World Driver's Championship 
                         \u2014 Results for each Race</h2>""")
    subtitle = Div(text="<i>Green coloring indicates top 10, regardless of the scoring system used this season.</i>")
    table = bokeh.layouts.row([Div(text=htmlcode)])

    return column([title, subtitle, table]), driver_source, constructor_source


def generate_dnf_table(year_results):
    """
    Generates a table showing the number of DNFs each constructor and driver has
    :param year_results: Year results
    :return: Table layout
    """
    def format_dnfs(finished, crashed, mechanical):
        total = finished + crashed + mechanical
        if total <= 0.01:
            return "0 / 0 (0%)", "0 / 0 (0%)", "0 / 0 (0%)"
        finished_str = f"{finished} / {total} ({round(100 * finished / total, 1)}%)".rjust(15)
        crashed_str = f"{crashed} / {total} ({round(100 * crashed / total, 1)}%)".rjust(15)
        mechanical_str = f"{mechanical} / {total} ({round(100 * mechanical / total, 1)}%)".rjust(15)
        return finished_str, mechanical_str, crashed_str

    constructor_source = pd.DataFrame(columns=["name", "finished_str", "mechanical_str", "crash_str"])
    driver_source = pd.DataFrame(columns=["name", "finished_str", "mechanical_str", "crash_str"])
    for constructor_id in year_results["constructorId"].unique():
        constructor_results = year_results[year_results["constructorId"] == constructor_id]
        constructor_name = get_constructor_name(constructor_id)
        classifications = constructor_results["statusId"].apply(get_status_classification)
        constructor_num_finished = classifications[classifications == "finished"].shape[0]
        constructor_num_crash_dnfs = classifications[classifications == "crash"].shape[0]
        constructor_num_mechanical_dnfs = classifications[classifications == "mechanical"].shape[0]
        formatted = format_dnfs(constructor_num_finished, constructor_num_crash_dnfs,
                                constructor_num_mechanical_dnfs)
        constructor_source = constructor_source.append({
            "name": constructor_name,
            "finished_str": formatted[0],
            "mechanical_str": formatted[1],
            "crash_str": formatted[2]
        }, ignore_index=True)
    for driver_id in year_results["driverId"].unique():
        driver_results = year_results[year_results["driverId"] == driver_id]
        driver_name = get_driver_name(driver_id)
        classifications = driver_results["statusId"].apply(get_status_classification)
        driver_num_finished = classifications[classifications == "finished"].shape[0]
        driver_num_crash_dnfs = classifications[classifications == "crash"].shape[0]
        driver_num_mechanical_dnfs = classifications[classifications == "mechanical"].shape[0]
        formatted = format_dnfs(driver_num_finished, driver_num_crash_dnfs, driver_num_mechanical_dnfs)
        driver_source = driver_source.append({
            "name": driver_name,
            "finished_str": formatted[0],
            "mechanical_str": formatted[1],
            "crash_str": formatted[2]
        }, ignore_index=True)

    constructor_title = Div(text=u"""<h2 style="margin-bottom:0px;">DNF Chart \u2014 Constructors</h2>""")
    driver_title = Div(text=u"""<h2 style="margin-bottom:0px;">DNF Chart \u2014 Drivers</h2>""")
    subtitle = Div(text="<i>Finished shows number of started and finished races, crashes include self-enforced errors,"
                        " the denominator of the fraction is races entered. DNFs are included if the driver did not "
                        "finish the Grand Prix, regardless of if they completed 90% of the race.</i>")

    dnf_columns = [
        TableColumn(field="name", title="Name", width=200),
        TableColumn(field="finished_str", title="Finished", width=150),
        TableColumn(field="mechanical_str", title="Mechanical DNFs", width=150),
        TableColumn(field="crash_str", title="Crash DNFs", width=150),
    ]
    driver_dnf_table = DataTable(source=ColumnDataSource(data=driver_source), columns=dnf_columns,
                                 index_position=None, min_height=530)
    constructor_dnf_table = DataTable(source=ColumnDataSource(data=constructor_source), columns=dnf_columns,
                                      index_position=None, min_height=530)
    return row([column([driver_title, subtitle, driver_dnf_table], sizing_mode="stretch_width"),
                column([constructor_title, subtitle, constructor_dnf_table], sizing_mode="stretch_width")],
               sizing_mode="stretch_width")


def generate_wcc_results_table(year_results, year_races, year_constructor_standings):
    """
    Summary table of WCC results, including:
    WCC finish pos
    Constructor name
    Drivers
    Points
    Num races maybe
    Wins (and %)
    Podiums (and %)
    Mean finish pos.
    :return:
    """
    logging.info("Generating WCC results table")
    source = pd.DataFrame(columns=["wcc_position",
                                   "constructor_name",
                                   "driver_names",
                                   "points",
                                   "wins",
                                   "podiums",
                                   "avg_fp"])
    last_rid = year_races.index.max()
    for cid in year_results["constructorId"].unique():
        constructor_constructor_standings = year_constructor_standings[
            (year_constructor_standings["constructorId"] == cid) &
            (year_constructor_standings["raceId"] == last_rid)]
        if constructor_constructor_standings.shape[0] > 0:
            wcc_position = int_to_ordinal(constructor_constructor_standings["position"].values[0])
            constructor_name = get_constructor_name(cid)
            constructor_results = year_results[year_results["constructorId"] == cid]
            driver_names = ", ".join([get_driver_name(did, include_flag=False, just_last=True)
                                      for did in constructor_results["driverId"].unique()])
            points = constructor_constructor_standings["points"].values[0]
            points = int(points) if abs(int(points) - points) < 0.01 and not np.isnan(points) else points
            points = str(points).rjust(3)
            wins = constructor_results[constructor_results["position"] == 1].shape[0]
            podiums = constructor_results[constructor_results["position"] <= 3].shape[0]
            num_races = constructor_results["raceId"].unique().shape[0]
            if num_races > 0:
                wins_str = str(wins) + " (" + str(round(100 * wins / num_races, 1)) + "%)"
                podiums_str = str(podiums) + " (" + str(round(100 * podiums / num_races, 1)) + "%)"
            else:
                wins_str = "0 (0.0%)"
                podiums_str = "0 (0.0%)"
            wins_str = wins_str.rjust(11)
            podiums_str = podiums_str.rjust(11)
            avg_fp = str(round(constructor_results["positionOrder"].mean(), 1)).rjust(4)
            source = source.append({
                "wcc_position": wcc_position,
                "constructor_name": constructor_name,
                "driver_names": driver_names,
                "points": points,
                "wins": wins_str,
                "podiums": podiums_str,
                "avg_fp": avg_fp
            }, ignore_index=True)
    source = source.sort_values(by="wcc_position")
    results_columns = [
        TableColumn(field="wcc_position", title="Pos.", width=30),
        TableColumn(field="constructor_name", title="Name", width=110),
        TableColumn(field="driver_names", title="Drivers", width=130),
        TableColumn(field="points", title="Pts.", width=30),
        TableColumn(field="wins", title="Wins", width=60),
        TableColumn(field="podiums", title="Podiums", width=50),
        TableColumn(field="avg_fp", title="Avg. Finish Pos.", width=80),
    ]

    title = Div(text=u"<h2>World Constructor's Championship</h2>")
    wcc_results_table = DataTable(source=ColumnDataSource(data=source), columns=results_columns, index_position=None)
    return column([title, wcc_results_table], sizing_mode="stretch_width")


def generate_error_layout():
    """
    Creates an error layout in the case where the year is not valid.
    :return: The error layout
    """
    logging.info("Generating error layout")
    text = "Somehow, you have selected an invalid season. The seasons we have data on are..."
    text += "<ul>"
    for year in seasons["year"].unique():
        text += f"<li>{str(year)}</li>"
    text += "</ul><br>"
    return Div(text=text)
