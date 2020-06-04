import logging
import math
import pandas as pd
import numpy as np
from bokeh.layouts import column, row
from bokeh.models import Div, Spacer, Span, Label, Title, Slider, ColumnDataSource, Range1d, LegendItem, Legend, \
    FixedTicker, CrosshairTool, HoverTool, LabelSet
from bokeh.plotting import figure
from pandas import Series
from data_loading.data_loader import load_drivers, load_results, load_driver_standings, load_races, \
    load_fastest_lap_data, load_constructor_standings
from mode import driver, constructor
from utils import get_constructor_name, get_driver_name, PLOT_BACKGROUND_COLOR, rounds_to_str, int_to_ordinal, \
    position_text_to_str, get_race_name, result_to_str, plot_image_url, vdivider

# Note, DC stands for driverconstructor

drivers = load_drivers()
results = load_results()
driver_standings = load_driver_standings()
constructor_standings = load_constructor_standings()
races = load_races()
fastest_lap_data = load_fastest_lap_data()


def get_layout(driver_id=-1, constructor_id=-1, download_image=True, **kwargs):
    # Check that the combo is valid
    constructor_results = results[results["constructorId"] == constructor_id]
    dc_results = constructor_results[constructor_results["driverId"] == driver_id]
    if dc_results.shape[0] == 0:
        return generate_error_layout(driver_id, constructor_id)
    dc_rids = dc_results["raceId"].unique()
    driver_driver_standings = driver_standings[driver_standings["driverId"] == driver_id]
    dc_driver_standings = driver_driver_standings[driver_driver_standings["raceId"].isin(dc_rids)]
    dc_races = races[races.index.isin(dc_rids)].sort_values(by=["year", "raceId"])
    dc_years = dc_races["year"].unique()
    dc_fastest_lap_data = fastest_lap_data[(fastest_lap_data["driver_id"] == driver_id) &
                                           (fastest_lap_data["raceId"].isin(dc_rids))]
    constructor_constructor_standings = constructor_standings[constructor_standings["constructorId"] == constructor_id]

    logging.info(f"Generating layout for mode DRIVERCONSTRUCTOR in driverconstructor, "
                 f"driver_id={driver_id}, constructor_id={constructor_id}")

    # Positions plot
    positions_plot, positions_source = generate_positions_plot(dc_years, dc_driver_standings, dc_results,
                                                               dc_fastest_lap_data, driver_id, constructor_id)
    mark_teammate_changes(positions_source, constructor_results, driver_id, positions_plot)

    # Win plot
    win_plot = generate_win_plot(positions_source)

    # Teammate finish pos vs driver finish pos scatter
    teammatefp_fp_scatter = generate_teammatefp_fp_scatter(positions_source, constructor_results, driver_id)

    # Teammate diff plot
    teammate_diff_plot, explanation_div, teammate_diff_source = generate_teammate_diff_comparison_scatter(
        positions_source, constructor_results, driver_id)

    # Teammate finish pos vs driver finish pos line plot
    slider, teammate_comparison_line, source = generate_teammate_comparison_line_plot(positions_source,
                                                                                      constructor_results, driver_id,
                                                                                      return_components_and_source=True)
    mark_teammate_changes(positions_source, constructor_results, driver_id, teammate_comparison_line)
    teammate_comparison_line = column([slider, teammate_comparison_line], sizing_mode="stretch_width")

    # Finishing position bar plot
    finishing_position_bar_plot = generate_finishing_position_bar_plot(dc_results)

    # WDC position bar plot
    wdc_position_bar_plot = generate_wdc_position_bar_plot(positions_source)

    # WCC position bar plot
    wcc_position_bar_plot, wcc_position_source = generate_wcc_position_bar_plot(dc_years,
                                                                                constructor_constructor_standings)

    # Start pos. vs finish pos. scatter
    spvfp_scatter = generate_spvfp_scatter(dc_results, dc_races, driver_driver_standings)

    # Mean lap time rank vs finish pos. scatter
    mltr_fp_scatter = generate_mltr_fp_scatter(dc_results, dc_races, driver_driver_standings, driver_id)

    # Circuit performance table
    circuit_performance_table = generate_circuit_performance_table(dc_results, dc_races, driver_id, constructor_id)

    # Stats
    stats_layout = generate_stats_layout(dc_years, dc_races, dc_results, positions_source, wcc_position_source,
                                         teammate_diff_source, driver_id, constructor_id)

    # Driver image
    # TODO make this an image of this driver at this constructor
    if download_image:
        image_url = str(drivers.loc[driver_id, "imgUrl"])
        image_view = plot_image_url(image_url)
    else:
        image_view = Div()

    header = get_driver_name(driver_id) + " at "
    header += get_constructor_name(constructor_id)
    header += " (" + rounds_to_str(dc_years) + ")"
    header = Div(text=f"<h2><b>{header}</b></h2>")

    middle_spacer = Spacer(width=5, background=PLOT_BACKGROUND_COLOR)
    divider = vdivider()
    layout = column([header,
                     positions_plot, middle_spacer,
                     win_plot, middle_spacer,
                     row([teammatefp_fp_scatter, teammate_diff_plot], sizing_mode="stretch_width"),
                     explanation_div, middle_spacer,
                     teammate_comparison_line, middle_spacer,
                     finishing_position_bar_plot, middle_spacer,
                     wdc_position_bar_plot, middle_spacer,
                     wcc_position_bar_plot, middle_spacer,
                     row([spvfp_scatter, mltr_fp_scatter], sizing_mode="stretch_width"),
                     circuit_performance_table,
                     row([image_view, divider, stats_layout], sizing_mode="stretch_both")],
                    sizing_mode="stretch_width")

    logging.info("Finished generating layout for mode DRIVERCONSTRUCTOR")

    return layout


def generate_positions_plot(dc_years, dc_driver_standings, dc_results, dc_fastest_lap_data, driver_id, constructor_id):
    """
    Plot WDC position (both rounds and full season), quali, fastest lap, and finishing position rank vs time all on the
    same graph along with smoothed versions of the quali, fastest lap, and finish position ranks.
    This method simply aliases `driver.generate_positions_plot`.
    :param dc_years: DC years
    :param dc_driver_standings: DC driver standings
    :param dc_results: DC results
    :param dc_fastest_lap_data: DC fastest lap data
    :param driver_id: Driver ID
    :param constructor_id: Constructor ID
    :return: Positions plot layout, positions source
    """
    title = get_driver_name(driver_id) + " at "
    title += get_constructor_name(constructor_id)
    title += " (" + rounds_to_str(dc_years) + ")"
    positions_plot, positions_source = driver.generate_positions_plot(dc_years, dc_driver_standings, dc_results,
                                                                      dc_fastest_lap_data, driver_id, title=title)
    subtitle = "Teammates who stayed for more than five races are marked with a vertical line."
    positions_plot.add_layout(Title(text=subtitle, text_font_style="italic"), "above")
    return positions_plot, positions_source


def mark_teammate_changes(positions_source, constructor_results, driver_id, fig):
    """
    Marks teammate changes with a vertical line (only teammates who raced for > 5 races).
    :param positions_source: Positions source
    :param constructor_results: Constructor results
    :param driver_id: Driver ID
    :param fig: Figure
    :return: Figure with teammate changes marked (also modifies in place)
    """
    prev_teammate_did = -1
    label_kwargs = dict(render_mode="canvas",
                        text_color="white",
                        text_font_size="12pt",
                        angle=math.pi / 4)
    for idx, row in positions_source.iterrows():
        if row["grid"] == "":
            continue
        rid = row["race_id"]
        dids = constructor_results[constructor_results["raceId"] == rid]["driverId"].unique().tolist()
        if driver_id in dids:
            dids.remove(driver_id)
        if len(dids) > 0:
            teammate_did = dids[0]
            if teammate_did != prev_teammate_did:
                teammate_results = constructor_results[constructor_results["driverId"] == teammate_did]
                if teammate_results.shape[0] > 5:
                    x = row["x"]
                    line = Span(line_color="white", location=x, dimension="height", line_alpha=0.4, line_width=3.2)
                    fig.add_layout(line)
                    label = Label(x=x + 0.1, y=18, text=get_driver_name(teammate_did, include_flag=False, just_last=True),
                                  **label_kwargs)
                    fig.add_layout(label)
            prev_teammate_did = teammate_did
    return fig


def generate_spvfp_scatter(dc_results, dc_races, driver_driver_standings):
    """
    Plot a scatter of quali position vs finish position and draw the y=x line
    :param dc_results: CD results
    :param dc_races: CD races
    :param driver_driver_standings: Driver driver standings
    :return: Start pos. vs finish pos. scatter layout
    """
    return driver.generate_spvfp_scatter(dc_results, dc_races, driver_driver_standings)


def generate_mltr_fp_scatter(dc_results, dc_races, driver_driver_standings, driver_id):
    """
    Plot scatter of mean lap time rank (x) vs finish position (y) to get a sense of what years the driver out-drove the
    car
    :param dc_results: DC results
    :param dc_races: DC races
    :param driver_driver_standings: Driver driver standings
    :param driver_id: Driver ID
    :return: Mean lap time rank vs finish position scatter plot layout
    """
    return driver.generate_mltr_fp_scatter(dc_results, dc_races, driver_driver_standings)


def generate_circuit_performance_table(dc_results, dc_races, driver_id, constructor_id):
    """
    Generates a table of the driver's performance at every circuit, ranked by number of wins then number of 2nd places,
    then number of 3rds, etc.
    :param dc_results: DC results
    :param dc_races: DC races
    :param driver_id: Driver ID
    :param constructor_id: Constructor ID
    :return: Circuit performance table layout, source
    """
    title = f"<h2><b>What were {get_driver_name(driver_id, include_flag=False, just_last=True)}'s Best Circuits with " \
            f"{get_constructor_name(constructor_id, include_flag=False)}?</b></h2>"
    return driver.generate_circuit_performance_table(dc_results, dc_races, driver_id, title=title)


def generate_win_plot(positions_source):
    """
    Plots number of races, win percentage, number of wins, podium percentage, number of podiums, top 6 percentage, and
    number of top 6s on the same plot (2 different axes on each side).
    :param positions_source: Positions source
    :return: Win plot layout
    """
    return driver.generate_win_plot(positions_source)


def generate_finishing_position_bar_plot(dc_results):
    """
    Bar plot showing distribution of race finishing positions.
    :param dc_results: DC results
    :return: Finish position bar plot layout
    """
    return driver.generate_finishing_position_bar_plot(dc_results, plot_height=300)


def generate_wdc_position_bar_plot(positions_source):
    """
    Bar plot showing distribution of WDC finishing positions.
    :param positions_source: Positions source
    :return: WDC position bar plot layout
    """
    return driver.generate_wdc_position_bar_plot(positions_source, plot_height=300)


def generate_wcc_position_bar_plot(dc_years, constructor_constructor_standings):
    """
    Bar plot showing distribution of WCC finishing positions.
    :param dc_years: DC years
    :param constructor_constructor_standings: Constructor constructor standings
    :return: WCC position bar plot layout
    """
    source = pd.DataFrame(columns=["year", "wcc_final_standing"])
    for year in dc_years:
        year_races = races[races["year"] == year]
        final_rid = year_races[year_races["round"] == year_races["round"].max()].index.values[0]
        final_standing = constructor_constructor_standings[constructor_constructor_standings["raceId"] == final_rid]
        final_standing = final_standing["position"]
        if final_standing.shape[0] > 0:
            final_standing = final_standing.values[0]
            source = source.append({
                "year": year,
                "wcc_final_standing": final_standing
            }, ignore_index=True)
    return constructor.generate_wcc_position_bar_plot(source, plot_height=300, color="orange"), source


def generate_teammatefp_fp_scatter(positions_source, constructor_results, driver_id, include_year_labels=False,
                                   include_race_labels=False):
    """
    Scatter plot of teammate finish position vs driver finish position along with a y=x line to show if he is beating
    his teammate
    :param positions_source: Positions source
    :param constructor_results: Constructor results
    :param driver_id: Driver ID
    :param include_year_labels: Whether to include year labels on the plot
    :param include_race_labels: Whether to include race labels on the plot
    :return: Teammate finish pos vs driver finish pos scatter
    """
    logging.info("Generating teammate finish pos. vs driver finish pos scatter")
    source = pd.DataFrame(columns=["driver_fp", "teammate_fp",
                                   "color", "marker", "size", "driver_fp_str", "teammate_fp_str",
                                   "year", "roundNum", "roundName", "roundFlag", "teammate_name"])
    prev_teammate_did = -1
    teammate_name = ""
    for idx, row in positions_source.iterrows():
        rid = row["race_id"]
        race_results = constructor_results[constructor_results["raceId"] == rid]
        dids = race_results["driverId"].unique().tolist()
        if driver_id in dids:
            dids.remove(driver_id)
        if len(dids) > 0:
            teammate_did = dids[0]
            if teammate_did != prev_teammate_did:
                teammate_name = get_driver_name(teammate_did)
                prev_teammate_did = teammate_did
            driver_fp = row["finish_position_int"]
            teammate_results = race_results[race_results["driverId"] == teammate_did]
            teammate_classification = position_text_to_str(teammate_results["positionText"].values[0])
            if teammate_results.shape[0] > 0:
                teammate_fp = teammate_results["positionOrder"].values[0]
                classification = row["finish_position_str"]
                if classification == "RET":
                    driver_fp_str = "RET"
                    marker = "x"
                    color = "red"
                    size = 10
                else:
                    driver_fp_str = int_to_ordinal(row["finish_position_int"])
                    if teammate_classification.isnumeric():
                        marker = "circle"
                        size = 8
                        color = "white"
                    else:  # RET or DNQ or something like that
                        marker = "diamond"
                        size = 12
                        color = "yellow"
                if teammate_classification.isnumeric():
                    teammate_fp_str = int_to_ordinal(int(teammate_classification))
                else:
                    teammate_fp_str = teammate_classification
                source = source.append({
                    "driver_fp": driver_fp,
                    "teammate_fp": teammate_fp,
                    "marker": marker,
                    "color": color,
                    "size": size,
                    "roundNum": row["roundNum"],
                    "roundName": row["roundName"],
                    "roundFlag": row["roundName"][:2],
                    "year": row["year"],
                    "driver_fp_str": driver_fp_str,
                    "teammate_fp_str": teammate_fp_str,
                    "teammate_name": teammate_name
                }, ignore_index=True)

    driver_name = get_driver_name(driver_id, include_flag=False, just_last=True)
    teammatefp_fp_scatter = figure(title="Teammate Finish Position vs Driver Finish Position",
                                   x_axis_label="Teammate Finish Position (Official Classification)",
                                   y_axis_label=f"{driver_name} Finish Position (Official Classification)",
                                   x_range=Range1d(0, 22, bounds=(0, 60)),
                                   y_range=Range1d(0, 22, bounds=(0, 60)),
                                   tools="pan,box_zoom,reset,save")
    teammatefp_fp_scatter.xaxis.ticker = FixedTicker(ticks=np.arange(5, 61, 5).tolist() + [1])
    teammatefp_fp_scatter.yaxis.ticker = FixedTicker(ticks=np.arange(5, 61, 5).tolist() + [1])

    teammatefp_fp_scatter.scatter(x="teammate_fp", y="driver_fp", source=source,
                                  marker="marker", color="color", size="size")
    teammatefp_fp_scatter.line(x=[-60, 60], y=[-60, 60], color="white", line_alpha=0.5)

    # Labels
    label_kwargs = dict(render_mode="canvas",
                        text_color="white",
                        text_font_size="12pt",
                        border_line_color="white",
                        border_line_alpha=0.7)
    label1 = Label(x=1, y=21, text=" Teammate Finished Higher ", **label_kwargs)
    label2 = Label(x=12, y=0.3, text=f" {driver_name} Finished Higher ", **label_kwargs)
    teammatefp_fp_scatter.add_layout(label1)
    teammatefp_fp_scatter.add_layout(label2)

    marker_label_kwargs = dict(x="teammate_fp",
                               y="driver_fp",
                               level="glyph",
                               x_offset=1.1,
                               y_offset=1.1,
                               source=ColumnDataSource(source),
                               render_mode="canvas",
                               text_color="white",
                               text_font_size="10pt")
    if include_year_labels:
        labels = LabelSet(text="year", **marker_label_kwargs)
        teammatefp_fp_scatter.add_layout(labels)
    if include_race_labels:
        labels = LabelSet(text="roundFlag", **marker_label_kwargs)
        teammatefp_fp_scatter.add_layout(labels)

    # Hover tooltip
    teammatefp_fp_scatter.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Year", "@year"),
        ("Finishing Position", "@driver_fp_str"),
        ("Teammate", "@teammate_name"),
        ("Teammate Finishing Position", "@teammate_fp_str"),
        ("Round", "@roundNum - @roundName")
    ]))

    # Crosshair tooltip
    teammatefp_fp_scatter.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

    return teammatefp_fp_scatter


def generate_teammate_diff_comparison_scatter(positions_source, constructor_results, driver_id,
                                              include_year_labels=False, include_race_labels=False):
    """
    Try a scatter plot where the x axis is mean lap time rank difference from teammate (his avg time rank - teammate's),
    y axis is position difference from teammate
    For example:
    Dots in +x and +y represents drives when he went slower than his teammate but finished worse than teammate
    Dots in -x +y represents drives where he went faster on average but still finished lower, possibly showing bias
        (or inferiority)
    :param positions_source: Positions source
    :param constructor_results: Constructor results
    :param driver_id: Driver ID
    :param include_year_labels: Whether to include year labels on the plot
    :param include_race_labels: Whether to include race labels on the plot
    :return: Teammate diff. comparison scatter, explanation div
    """
    # TODO make this plot use mean lap time percent rather than rank
    logging.info("Generating teammate diff comparison scatter")
    source = pd.DataFrame(columns=["mltr_diff", "mlt_diff", "fp_diff",
                                   "driver_mltr_str", "teammate_mltr_str",
                                   "driver_fp_str", "teammate_fp_str",
                                   "color", "marker", "size",
                                   "year", "roundNum", "roundName", "roundFlag", "teammate_name"])
    prev_teammate_did = -1
    teammate_name = ""
    for idx, row in positions_source.iterrows():
        rid = row["race_id"]
        race_results = constructor_results[constructor_results["raceId"] == rid]
        dids = race_results["driverId"].unique().tolist()
        if driver_id in dids:
            dids.remove(driver_id)
        if len(dids) > 0:
            teammate_did = dids[0]
            if teammate_did != prev_teammate_did:
                teammate_name = get_driver_name(teammate_did)
                prev_teammate_did = teammate_did
            driver_fp = row["finish_position_int"]
            teammate_results = race_results[race_results["driverId"] == teammate_did]
            if teammate_results.shape[0] > 0:
                teammate_fp = teammate_results["positionOrder"].values[0]
                fp_diff = driver_fp - teammate_fp
                race_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"] == rid]
                race_fastest_lap_data = race_fastest_lap_data.set_index("driver_id")
                avg_lap_ranks = race_fastest_lap_data["avg_lap_time_millis"].rank()
                if driver_id in avg_lap_ranks.index and teammate_did in avg_lap_ranks.index:
                    driver_mltr = avg_lap_ranks.loc[driver_id]
                    teammate_mltr = avg_lap_ranks.loc[teammate_did]
                    mltr_diff = driver_mltr - teammate_mltr
                    if isinstance(driver_mltr, Series):
                        driver_mltr = driver_mltr.values[0]
                    if isinstance(teammate_mltr, Series):
                        teammate_mltr = teammate_mltr.values[0]
                    driver_mltr_str = int_to_ordinal(driver_mltr)
                    teammate_mltr_str = int_to_ordinal(teammate_mltr)

                    driver_mlt = race_fastest_lap_data.loc[driver_id, "avg_lap_time_millis"]
                    teammate_mlt = race_fastest_lap_data.loc[teammate_did, "avg_lap_time_millis"]
                    mlt_diff = driver_mlt - teammate_mlt
                    if isinstance(mlt_diff, Series):
                        mlt_diff = mlt_diff.values[0]

                    teammate_classification = position_text_to_str(teammate_results["positionText"].values[0])
                    classification = row["finish_position_str"]
                    if classification == "RET":
                        driver_fp_str = "RET"
                        marker = "x"
                        color = "red"
                        size = 10
                    else:
                        driver_fp_str = int_to_ordinal(row["finish_position_int"])
                        if teammate_classification.isnumeric():
                            marker = "circle"
                            size = 8
                            color = "white"
                        else:  # RET or DNQ or something like that
                            marker = "diamond"
                            size = 12
                            color = "yellow"
                    if teammate_classification.isnumeric():
                        teammate_fp_str = int_to_ordinal(int(teammate_classification))
                    else:
                        teammate_fp_str = teammate_classification

                    source = source.append({
                        "mltr_diff": mltr_diff,
                        "fp_diff": fp_diff,
                        "driver_mltr_str": driver_mltr_str,
                        "teammate_mltr_str": teammate_mltr_str,
                        "teammate_name": teammate_name,
                        "roundNum": row["roundNum"],
                        "roundName": row["roundName"],
                        "roundFlag": row["roundName"][:2],
                        "year": row["year"],
                        "marker": marker,
                        "color": color,
                        "size": size,
                        "driver_fp_str": driver_fp_str,
                        "teammate_fp_str": teammate_fp_str,
                        "mlt_diff": mlt_diff
                    }, ignore_index=True)
    teammate_diff_scatter = figure(title="Teammate Average Lap Time Comparison (explanation below)",
                                   x_axis_label="Average Lap Time Rank Difference",
                                   y_axis_label="Finish Position Difference",
                                   x_range=Range1d(-22, 22, bounds=(-40, 40)),
                                   y_range=Range1d(-22, 22, bounds=(-40, 40)),
                                   tools="pan,box_zoom,reset,save")
    teammate_diff_scatter.xaxis.ticker = FixedTicker(ticks=np.arange(-45, 45, 5).tolist())
    teammate_diff_scatter.yaxis.ticker = FixedTicker(ticks=np.arange(-45, 45, 5).tolist())

    driver_name = get_driver_name(driver_id, include_flag=False, just_last=True)
    explanation = f"This plot is meant to show when {driver_name} was faster than his/her teammate, but yet finished " \
                  f"lower, and vice \nversa, along with the more regular cases of when {driver_name} was faster than " \
                  f"his/her teammate, and finished higher (and vice versa).<br>To compute the x axis, first every " \
                  f"driver in each race is ranked based on their average lap time (1 being fastest). Next, the x axis" \
                  f" value is computed for every race as {driver_name}'s fastest lap rank - teammate's " \
                  f"fastest lap rank.<br>The y axis is computed for every race as {driver_name}'s finishing position " \
                  f"- teammate's finishing position (official classification is used so therefore DNFs are plotted)." \
                  f"<br>Thus, dots in the +x region represent races where {driver_name} was slower than his/her " \
                  f"teammate overall and dots in the +y region represent races where {driver_name} finished lower " \
                  f"than his/her teammate.<br>Some may view this as a \"When/how often does {driver_name} " \
                  f"get screwed over\" chart."
    explanation = Div(text=explanation)

    teammate_diff_scatter.scatter(x="mltr_diff", y="fp_diff", source=source,
                                  color="color", size="size", marker="marker")
    teammate_diff_scatter.line(x=[-60, 60], y=[-60, 60], color="white", line_alpha=0.2)
    teammate_diff_scatter.line(x=[-60, 60], y=[0, 0], color="white", line_alpha=0.5, line_width=1.2)
    teammate_diff_scatter.line(x=[0, 0], y=[-60, 60], color="white", line_alpha=0.5, line_width=1.2)

    # Labels
    label_kwargs = dict(render_mode="canvas",
                        text_color="white",
                        text_font_size="8pt",
                        border_line_color="white",
                        border_line_alpha=0.7)
    label1 = Label(x=5, y=20, text=" Teammate Faster and Finished Higher ", **label_kwargs)
    label2 = Label(x=-21, y=20, text=f" {driver_name} Faster but Teammate Finished Higher ", **label_kwargs)
    label3 = Label(x=-19, y=-20, text=f" {driver_name} Faster and Finished Higher ", **label_kwargs)
    label4 = Label(x=1, y=-20, text=f" Teammate Faster but {driver_name} Finished Higher ", **label_kwargs)
    teammate_diff_scatter.add_layout(label1)
    teammate_diff_scatter.add_layout(label2)
    teammate_diff_scatter.add_layout(label3)
    teammate_diff_scatter.add_layout(label4)

    marker_label_kwargs = dict(x="mltr_diff",
                               y="fp_diff",
                               level="glyph",
                               x_offset=1.1,
                               y_offset=1.1,
                               source=ColumnDataSource(source),
                               render_mode="canvas",
                               text_color="white",
                               text_font_size="10pt")
    if include_year_labels:
        labels = LabelSet(text="year", **marker_label_kwargs)
        teammate_diff_scatter.add_layout(labels)
    if include_race_labels:
        labels = LabelSet(text="roundFlag", **marker_label_kwargs)
        teammate_diff_scatter.add_layout(labels)

    # Hover tooltip
    teammate_diff_scatter.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Year", "@year"),
        ("Finishing Position", "@driver_fp_str"),
        ("Teammate Finishing Position", "@teammate_fp_str"),
        ("Avg. Lap Time Rank", "@driver_mltr_str"),
        ("Teammate Avg. Lap Time Rank", "@teammate_mltr_str"),
        ("Teammate", "@teammate_name"),
        ("Round", "@roundNum - @roundName")
    ]))

    # Crosshair tooltip
    teammate_diff_scatter.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

    return teammate_diff_scatter, explanation, source


def generate_teammate_comparison_line_plot(positions_source, constructor_results, driver_id,
                                           return_components_and_source=False, default_alpha=0.1, mute_smoothed=False):
    """
    Line plot of driver and teammate finish position vs time
    :param positions_source: Positions source
    :param constructor_results: Constructor results
    :param driver_id: Driver ID
    :param return_components_and_source: If True, will return the slider, figure, source as a tuple, if False will
    return full layout
    :param default_alpha: Default alpha for smoothing
    :param mute_smoothed: If True, will mute the smoothed lines, if False will mute unsmoothed
    :return: Teammate comparison line plot layout or a tuple of slider, figure depending on `return_components`
    """
    # TODO add mean lap time percent to this plot
    logging.info("Generating teammate finish pos. vs driver finish pos line plot")
    source = pd.DataFrame(columns=["x", "year", "race_id",
                                   "driver_fp", "teammate_fp",
                                   "driver_fp_str", "teammate_fp_str"
                                   "year", "roundNum", "roundName", "wdc_final_standing", "teammate_name"])
    prev_teammate_did = -1
    teammate_name = ""
    for idx, row in positions_source.iterrows():
        rid = row["race_id"]
        x = row["x"]
        race_results = constructor_results[constructor_results["raceId"] == rid]
        dids = race_results["driverId"].unique().tolist()
        if driver_id in dids:
            dids.remove(driver_id)
        if len(dids) > 0:
            teammate_did = dids[0]
            if teammate_did != prev_teammate_did:
                teammate_name = get_driver_name(teammate_did)
                prev_teammate_did = teammate_did
            driver_fp = row["finish_position_int"]
            driver_fp_str = row["finish_position_str"]
            teammate_results = race_results[race_results["driverId"] == teammate_did]
            if teammate_results.shape[0] > 0:
                teammate_fp = teammate_results["positionOrder"].values[0]
                teammate_status_id = teammate_results["statusId"].values[0]
                teammate_fp_str, _ = result_to_str(teammate_fp, teammate_status_id)
                source = source.append({
                    "race_id": rid,
                    "x": x,
                    "driver_fp": driver_fp,
                    "teammate_fp": teammate_fp,
                    "driver_fp_str": driver_fp_str,
                    "teammate_fp_str": teammate_fp_str,
                    "year": row["year"],
                    "roundNum": row["roundNum"],
                    "roundName": row["roundName"],
                    "wdc_final_standing": row["wdc_final_standing"],
                    "teammate_name": teammate_name
                }, ignore_index=True)

    source["fp_diff"] = source["driver_fp"] - source["teammate_fp"]

    source["driver_fp_smoothed"] = source["driver_fp"].ewm(alpha=default_alpha).mean()
    source["teammate_fp_smoothed"] = source["teammate_fp"].ewm(alpha=default_alpha).mean()

    column_source = ColumnDataSource(data=source)

    min_x = source["x"].min()
    max_x = source["x"].max()
    teammate_fp_plot = figure(title=u"Teammate Comparison Over Time (T.M. stands for teammate) \u2014 Horizontal lines "
                                    u"show mean finish position, include DNFs",
                              x_axis_label="Year",
                              y_axis_label="Finish Position",
                              x_range=Range1d(min_x, max_x, bounds=(min_x, max_x + 3)),
                              y_range=Range1d(0, 22, bounds=(0, 60)),
                              tools="pan,box_zoom,reset,save")
    teammate_fp_plot.xaxis.ticker = FixedTicker(ticks=np.arange(1950, 2050))
    teammate_fp_plot.yaxis.ticker = FixedTicker(ticks=np.arange(5, 61, 5).tolist() + [1])

    kwargs = dict(
        x="x",
        source=column_source,
        line_width=2,
        muted_alpha=0
    )
    driver_fp_line = teammate_fp_plot.line(y="driver_fp", color="white", **kwargs)
    driver_fp_smoothed_line = teammate_fp_plot.line(y="driver_fp_smoothed", color="white", line_dash="dashed", **kwargs)
    teammate_fp_line = teammate_fp_plot.line(y="teammate_fp", color="yellow", **kwargs)
    teammate_fp_smoothed_line = teammate_fp_plot.line(y="teammate_fp_smoothed", color="yellow", line_dash="dashed",
                                                      **kwargs)

    # Draw line at means
    mean_driver_fp = source["driver_fp"].mean()
    mean_teammate_fp = source["teammate_fp"].mean()
    line_kwargs = dict(
        x=[-1000, 5000],
        line_alpha=0.4,
        line_width=2.5,
        muted_alpha=0
    )
    driver_mean_line = teammate_fp_plot.line(line_color="white", y=[mean_driver_fp] * 2, **line_kwargs)
    teammate_mean_line = teammate_fp_plot.line(line_color="yellow", y=[mean_teammate_fp] * 2, **line_kwargs)

    if mute_smoothed:
        driver_fp_smoothed_line.muted = True
        teammate_fp_smoothed_line.muted = True
    else:
        driver_fp_line.muted = True
        teammate_fp_line.muted = True
        driver_mean_line.muted = True
        teammate_mean_line.muted = True

    # Legend
    legend = [LegendItem(label="Driver Finish Pos.", renderers=[driver_fp_line, driver_mean_line]),
              LegendItem(label="Finish Pos. Smoothed", renderers=[driver_fp_smoothed_line]),
              LegendItem(label="Teammate Finish Pos.", renderers=[teammate_fp_line, teammate_mean_line]),
              LegendItem(label="T.M. Finish Pos. Smoothed", renderers=[teammate_fp_smoothed_line])]
    legend = Legend(items=legend, location="top_right", glyph_height=15, spacing=2, inactive_fill_color="gray")
    teammate_fp_plot.add_layout(legend, "right")
    teammate_fp_plot.legend.click_policy = "mute"
    teammate_fp_plot.legend.label_text_font_size = "12pt"

    # Smoothing slider
    def smoothing_cb(new):
        alpha = 1 - new
        if alpha < 0.01:
            alpha = 0.01
        if alpha > 0.99:
            source["driver_fp_smoothed"] = source["driver_fp"]
            source["teammate_fp_smoothed"] = source["teammate_fp"]
        else:
            source["driver_fp_smoothed"] = source["driver_fp"].ewm(alpha=alpha).mean()
            source["teammate_fp_smoothed"] = source["teammate_fp"].ewm(alpha=alpha).mean()
        column_source.patch({
            "driver_fp_smoothed": [(slice(source["driver_fp_smoothed"].shape[0]), source["driver_fp_smoothed"])],
            "teammate_fp_smoothed": [(slice(source["teammate_fp_smoothed"].shape[0]), source["teammate_fp_smoothed"])],
        })

    smoothing_slider = Slider(start=0, end=1, value=1 - default_alpha, step=.01, title="Smoothing Amount, 0=no "
                                                                                       "smoothing, 1=heavy smoothing")
    smoothing_slider.on_change("value", lambda attr, old, new: smoothing_cb(new))

    # Hover tooltip
    teammate_fp_plot.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Finish Position", "@driver_fp_str"),
        ("Teammate", "@teammate_name"),
        ("Teammate Finish Position", "@teammate_fp_str"),
        ("Year", "@year"),
        ("Round", "@roundNum - @roundName"),
        ("Final Position this year", "@wdc_final_standing")
    ]))

    # Crosshair tooltip
    teammate_fp_plot.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

    if return_components_and_source:
        return smoothing_slider, teammate_fp_plot, source
    else:
        return column([smoothing_slider, teammate_fp_plot], sizing_mode="stretch_width")


def generate_stats_layout(dc_years, dc_races, dc_results, positions_source, wcc_position_source, teammate_diff_source,
                          driver_id, constructor_id):
    """
    Career at this constructor summary div, including:
    - Highest finish in WDC while at this constructor
    - Highest WCC at this constructor
    - Race debut for this constructor
    - Last race for this constructor
    - Number of races entered for this constructor
    - Points scored for this constructor
    - Mean grid position
    - Mean finish position
    - Teammates
    - Mean gap to teammate in positions
    - Mean gap to teammate in time
    :param dc_years: DC Years
    :param dc_races: DC races
    :param dc_results: DC results
    :param positions_source: Positions source
    :param wcc_position_source: WCC positions source
    :param teammate_diff_source: Teammate diff. source
    :param driver_id: Driver ID
    :param constructor_id: Constructor ID
    :return: Stats div layout
    """
    logging.info("Generating driver stats layout")
    if dc_results.shape[0] == 0:
        return Div(text="")
    years_active = rounds_to_str(dc_years)

    first_year = np.min(dc_years)
    first_rid = dc_races[dc_races["year"] == first_year].sort_values(by="round").index.values[0]
    first_race_name = str(first_year) + " " + get_race_name(first_rid)
    last_year = np.max(dc_years)
    last_rid = dc_races[dc_races["year"] == last_year].sort_values(by="round", ascending=True).index.values[0]
    last_race_name = str(last_year) + " " + get_race_name(last_rid)

    wins = []
    driver_results = dc_results.set_index("raceId")
    for rid, race_row in dc_races.iterrows():
        position = driver_results.loc[rid, "positionOrder"]
        if isinstance(position, Series):
            position = position.values[0]
        if rid in driver_results.index and position == 1:
            wins.append(rid)
    if len(wins) > 0:
        first_win_rid = wins[0]
        first_win_year = races.loc[first_win_rid, "year"]
        first_win_name = str(first_win_year) + " " + get_race_name(first_win_rid)
        last_win_rid = wins[-1]
        last_win_year = races.loc[first_win_rid, "year"]
        last_win_name = str(last_win_year) + " " + get_race_name(last_win_rid)

    num_podiums = positions_source[positions_source["finish_position_int"] <= 3].shape[0]
    num_podiums_str = str(num_podiums)
    num_wins = positions_source[positions_source["finish_position_int"] == 1].shape[0]
    num_wins_str = str(num_wins)
    num_races = dc_results.shape[0]
    num_points = dc_results["points"].sum()
    num_points_str = str(num_points)
    if num_races > 0:
        num_podiums_str += " (" + str(round(num_podiums / num_races, 1)) + "%)"
        num_wins_str += " (" + str(round(num_wins / num_races, 1)) + "%)"
        num_points_str += " (" + str(round(num_points / num_races, 1)) + "points / race)"
    mean_sp = round(driver_results["grid"].mean(), 1)
    mean_fp = round(driver_results["positionOrder"].mean(), 1)
    teammates = ", ".join(teammate_diff_source["teammate_name"].unique())
    mean_teammate_gap_pos = round(teammate_diff_source["fp_diff"].mean(), 1)
    if mean_teammate_gap_pos < 0:
        mean_teammate_gap_pos = str(abs(mean_teammate_gap_pos)) + " places higher"
    else:
        mean_teammate_gap_pos = str(abs(mean_teammate_gap_pos)) + " places lower"
    if teammate_diff_source["mlt_diff"].isna().sum() == teammate_diff_source.shape[0]:
        mean_teammate_gap_str = ""
    else:
        mean_teammate_gap_time = teammate_diff_source["mlt_diff"].fillna(0.0).mean()
        mean_teammate_gap_str = mean_teammate_gap_pos
        if not np.isnan(mean_teammate_gap_time):
            if mean_teammate_gap_time < 0:
                mean_teammate_gap_time = str(abs(int(mean_teammate_gap_time))) + "ms faster"
            else:
                mean_teammate_gap_time = str(abs(int(mean_teammate_gap_time))) + "ms slower"
            mean_teammate_gap_str += " (" + mean_teammate_gap_time + ")"

    num_championships = 0
    championships_str = " ("
    years_hit = set()
    for idx, row in positions_source.iterrows():
        standing = row["wdc_final_standing"]
        year = row["year"]
        if year in years_hit:
            continue
        years_hit.add(year)
        if standing == 1:
            num_championships += 1
            championships_str += str(year) + ", "
    if num_championships == 0:
        championships_str = str(np.max(positions_source["wdc_final_standing"]))
    else:
        championships_str = str(num_championships) + championships_str[:-2] + ")"
    wcc_num_championships = 0
    wcc_str = " ("
    years_hit = set()
    for idx, row in wcc_position_source.iterrows():
        year = row["year"]
        standing = row["wcc_final_standing"]
        if year in years_hit:
            continue
        years_hit.add(year)
        if standing == 1:
            wcc_num_championships += 1
            wcc_str += str(year) + ", "
    if wcc_num_championships == 0:
        wcc_championships_str = str(np.max(wcc_position_source["wcc_final_standing"]))
    else:
        wcc_championships_str = str(wcc_num_championships) + wcc_str[:-2] + ")"

    header_template = """
    <h2 style="text-align: center;"><b>{}</b></h2>
    """

    template = """
    <pre><b>{}</b> {}<br></pre>
    """

    constructor_name = get_constructor_name(constructor_id)
    dc_stats = header_template.format(f"{get_driver_name(driver_id)}'s Stats at {constructor_name}")
    dc_stats += template.format("Years At Constructor: ".ljust(22), years_active)
    dc_stats += template.format("Entries: ".ljust(22), num_races)
    if num_championships == 0:
        dc_stats += template.format("Highest WDC Finish: ".ljust(22), championships_str)
    else:
        dc_stats += template.format("WDC Championships: ".ljust(22), championships_str)
    if num_championships == 0:
        dc_stats += template.format("Highest WCC Finish: ".ljust(22), wcc_championships_str)
    else:
        dc_stats += template.format("WCC Championships: ".ljust(22), wcc_championships_str)
    dc_stats += template.format("Wins: ".ljust(22), num_wins_str)
    dc_stats += template.format("Podiums: ".ljust(22), num_podiums_str)
    dc_stats += template.format("Points: ".ljust(22), num_points_str)
    dc_stats += template.format("Avg. Start Pos.: ".ljust(22), mean_sp)
    dc_stats += template.format("Avg. Finish Pos.: ".ljust(22), mean_fp)
    dc_stats += template.format("First Entry: ".ljust(22), first_race_name)
    if len(wins) > 0:
        dc_stats += template.format("First Win: ".ljust(22), first_win_name)
        dc_stats += template.format("Last Win: ".ljust(22), last_win_name)
    dc_stats += template.format("Last Entry: ".ljust(22), last_race_name)
    dc_stats += template.format("Teammates: ".ljust(22), teammates)
    if teammate_diff_source["mlt_diff"].isna().sum() != teammate_diff_source.shape[0]:
        dc_stats += template.format("Avg. Gap to Teammate: ".ljust(22), mean_teammate_gap_str)

    return Div(text=dc_stats)


def generate_error_layout(driver_id, constructor_id):
    """
    Generates an error layout in the event that the given driver never competed for the given constructor.
    :param driver_id: Driver ID
    :param constructor_id: Driver ID
    :return: Div layout
    """
    logging.info("Generating error layout")
    constructor_name = get_constructor_name(constructor_id, include_flag=False)
    driver_name = get_driver_name(driver_id, include_flag=False)
    constructor_results = results[results["constructorId"] == constructor_id]
    dids_for_that_constructor = constructor_results["driverId"].unique()
    cids_for_that_driver = results[results["driverId"] == driver_id]["constructorId"].unique()

    # Generate the text
    text = f"Unfortunately, {driver_name} never competed for {constructor_name}. Here are some other options:<br>"
    text += f"{driver_name} competed for the following constructors..."
    text += "<ul>"
    for cid in cids_for_that_driver:
        text += f"<li>{get_constructor_name(cid)}</li>"
    text += "</ul>"
    text += f"{constructor_name} has had the following drivers..."
    text += "<ul>"
    for did in dids_for_that_constructor:
        text += f"<li>{get_driver_name(did)}</li>"
    text += "</ul><br>"

    layout = Div(text=text)
    return layout


def is_valid_input(driver_id, constructor_id):
    """
    Returns whether the given input is a valid combination of driver and constructor. Used only for unit tests.
    :param driver_id: Driver ID
    :param constructor_id: Constructor ID
    :return: True if valid input, False otherwise
    """
    constructor_results = results[results["constructorId"] == constructor_id]
    dc_results = constructor_results[constructor_results["driverId"] == driver_id]
    return dc_results.shape[0] > 0
