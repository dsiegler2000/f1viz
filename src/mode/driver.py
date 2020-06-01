import logging
import math
from datetime import datetime
from collections import defaultdict, OrderedDict
from bokeh.palettes import Set3_12 as palette
import numpy as np
from bokeh.layouts import column, row
from bokeh.models import Div, FixedTicker, Range1d, CrosshairTool, HoverTool, LegendItem, Legend, Span, Label, Spacer, \
    ColumnDataSource, LinearAxis, NumeralTickFormatter, DataTable, TableColumn, FactorRange, LabelSet, Title, \
    DatetimeTickFormatter
import pandas as pd
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from pandas import Series

from mode import driverconstructor
from utils import get_driver_name, position_text_to_str, get_constructor_name, ColorDashGenerator, get_circuit_name, \
    PLOT_BACKGROUND_COLOR, int_to_ordinal, get_status_classification, rounds_to_str, get_race_name, nationality_to_flag, \
    result_to_str, millis_to_str, DATETIME_TICK_KWARGS, rescale, plot_image_url, vdivider
from data_loading.data_loader import load_drivers, load_results, load_driver_standings, load_races, \
    load_fastest_lap_data, load_status

drivers = load_drivers()
results = load_results()
driver_standings = load_driver_standings()
races = load_races()
fastest_lap_data = load_fastest_lap_data()
status = load_status()


def get_layout(driver_id=-1, download_image=True, **kwargs):
    if driver_id not in drivers.index:
        return generate_error_layout()

    driver_results = results[results["driverId"] == driver_id]
    driver_driver_standings = driver_standings[driver_standings["driverId"] == driver_id]
    driver_rids = driver_results["raceId"].unique()
    driver_races = races[races.index.isin(driver_rids)].sort_values(by=["year", "raceId"])
    driver_years = driver_races["year"].unique()
    driver_fastest_lap_data = fastest_lap_data[fastest_lap_data["driver_id"] == driver_id]
    constructor_results_idxs = []
    for idx, results_row in driver_results.iterrows():
        constructor_id = results_row["constructorId"]
        rid = results_row["raceId"]
        results_slice = results[(results["constructorId"] == constructor_id) &
                                (results["raceId"] == rid)]
        constructor_results_idxs.extend(results_slice.index.values.tolist())
    constructor_results = results.loc[constructor_results_idxs]

    logging.info(f"Generating layout for mode DRIVER in driver, driver_id={driver_id}")

    # Position plot
    positions_plot, positions_source = generate_positions_plot(driver_years, driver_driver_standings, driver_results,
                                                               driver_fastest_lap_data, driver_id)

    # Circuit performance table
    circuit_performance_table = generate_circuit_performance_table(driver_results, driver_races, driver_id)

    # Position bar plot
    position_dist = generate_finishing_position_bar_plot(driver_results)

    # WDC Position bar plot
    wdc_position_dist = generate_wdc_position_bar_plot(positions_source)

    # Win/podium plot
    win_plot = generate_win_plot(positions_source, driver_id=driver_id)

    # Starting position vs finish position scatter
    spvfp_scatter = generate_spvfp_scatter(driver_results, driver_races, driver_driver_standings)

    # Mean lap time rank vs finish position scatter plot
    mltr_fp_scatter = generate_mltr_fp_scatter(driver_results, driver_races, driver_driver_standings)

    # Teammate comparison line plot
    teammate_comparison_line_plot = generate_teammate_comparison_line_plot(positions_source, constructor_results,
                                                                           driver_years, driver_results, driver_id)

    # Team performance graph and table
    team_performance_layout, performance_source = generate_team_performance_layout(driver_races, positions_source,
                                                                                   driver_results)

    # Driver stats
    driver_stats_layout = generate_stats_layout(driver_years, driver_races, performance_source, driver_results,
                                                driver_id)

    # Header
    header = Div(text=f"<h2><b>{get_driver_name(driver_id)}</b></h2><br>")

    # Driver image
    if download_image:
        image_url = str(drivers.loc[driver_id, "imgUrl"])
        image_view = plot_image_url(image_url)
    else:
        image_view = Div()

    middle_spacer = Spacer(width=5, background=PLOT_BACKGROUND_COLOR)
    divider = vdivider()
    layout = column([header,
                     positions_plot, middle_spacer,
                     position_dist, middle_spacer,
                     wdc_position_dist, middle_spacer,
                     win_plot, middle_spacer,
                     row([spvfp_scatter, mltr_fp_scatter], sizing_mode="stretch_width"),
                     teammate_comparison_line_plot,
                     circuit_performance_table,
                     team_performance_layout,
                     row([image_view, divider, driver_stats_layout], sizing_mode="stretch_both")],
                    sizing_mode="stretch_width")

    logging.info("Finished generating layout for mode DRIVER")

    return layout


def generate_positions_plot(driver_years, driver_driver_standings, driver_results, driver_fastest_lap_data, driver_id,
                            title=None, smoothing_alpha=0.05, minor_line_width=1.7, major_line_width=2.8,
                            smoothing_muted=False, races_sublist=None, show_mean_finish_pos=False,
                            include_lap_times=False):
    """
    Plots WDC standing (calculated per-race and per-season), quali rank (per-race), fastest lap rank (per-race), and
    finishing position (per-race) all on the same plot.
    :param driver_years: Driver years
    :param driver_driver_standings: Driver driver standings
    :param driver_results: Driver results
    :param driver_fastest_lap_data: Driver fastest lap data
    :param driver_id: Driver ID
    :param title: Title override, if left None then will be the driver name
    :param smoothing_alpha: Alpha used for smoothing, 0.01=very smoothed, 0.99=almost no smoothing
    :param minor_line_width: Line width for the less important lines
    :param major_line_width: Line width for the more important lines
    :param smoothing_muted: If True, the smoothed lines will be muted by default, otherwise the main lines will be muted
    :param races_sublist: This is an option to only use a certain sub-set of races if, for example, the user wishes to
    show positions at just a specific circuit, set to None if looking at all races
    :param show_mean_finish_pos: If set to True, will show mean finish position that year instead of WDC finish pos
    :param include_lap_times: If set to True, will plot average lap time of every race
    :return: Plot layout, data source
    """
    # TODO add smoothing slider
    logging.info("Generating position plot")
    if driver_years.shape[0] == 0:
        return Div(text="Unfortunately, we don't have data on this driver."), {}
    source = pd.DataFrame(columns=[
        "x",
        "constructor_name",
        "finish_position_int", "finish_position_str",
        "race_id",
        "year",
        "grid", "grid_str",
        "avg_lap_rank", "avg_lap_rank_str",
        "points",
        "roundNum",
        "roundName",
        "wdc_current_standing", "wdc_current_standing_str",
        "wdc_final_standing", "wdc_final_standing_str",
        "avg_lap_time_millis", "avg_lap_time_str",
        "avg_finish_pos", "avg_finish_pos_str"
    ])
    min_year = driver_years.min()
    max_year = driver_years.max()
    constructor_name = "UNKNOWN"
    points = 0
    round_num = 1
    round_name = ""
    for year in range(min_year, max_year + 1):
        if races_sublist is None:
            year_subraces = races[races["year"] == year]
        else:
            year_subraces = races_sublist[races_sublist["year"] == year]
        if year_subraces.shape[0] == 0:
            continue
        year_races = races[races["year"] == year]
        year_driver_results = driver_results[driver_results["raceId"].isin(year_subraces.index)]
        year_fastest_lap_data = driver_fastest_lap_data[driver_fastest_lap_data["raceId"].isin(year_races.index)]
        avg_finish_pos = driver_results[driver_results["raceId"].isin(year_races.index)]["positionOrder"].mean()
        dx = 1 / year_subraces.shape[0]
        x = year
        final_rid = year_races[year_races["round"] == year_races["round"].max()].index.values[0]
        final_standing = driver_driver_standings[driver_driver_standings["raceId"] == final_rid]["position"]
        if final_standing.shape[0] == 0:
            final_standing = np.nan
            final_standing_str = ""
        else:
            final_standing = final_standing.values[0]
            final_standing_str = int_to_ordinal(final_standing)
        for race_id in year_subraces.sort_index().index:
            current_standing = driver_driver_standings[driver_driver_standings["raceId"] == race_id]
            race_results = year_driver_results[year_driver_results["raceId"] == race_id]
            if race_results.shape[0] > 0:
                finish_position_int = race_results["positionOrder"].astype(int).values[0]
                finish_position_str, _ = result_to_str(finish_position_int, race_results["statusId"].values[0])
                constructor_name = get_constructor_name(race_results["constructorId"].values[0])
                grid = race_results["grid"].values[0]
                grid_str = int_to_ordinal(grid)
            else:
                finish_position_str = "RET"
                finish_position_int = np.nan
                grid = np.nan
                grid_str = ""
            if current_standing.shape[0] and current_standing.shape[0] > 0:
                current_wdc_standing = current_standing["position"].values[0]
                current_wdc_standing_str = int_to_ordinal(current_wdc_standing)
                round_num = current_standing["roundNum"].values[0]
                round_name = current_standing["roundName"].values[0]
                points = current_standing["points"].values[0]
            else:
                current_wdc_standing = 20
                current_wdc_standing_str = ""
            race_fastest_lap_data = year_fastest_lap_data[year_fastest_lap_data["raceId"] == race_id]
            avg_lap_rank = race_fastest_lap_data["avg_lap_time_rank"]
            if avg_lap_rank.shape[0] > 0:
                avg_lap_rank = float(avg_lap_rank.values[0])
                if np.isnan(avg_lap_rank):
                    avg_lap_rank_str = ""
                else:
                    avg_lap_rank_str = int_to_ordinal(int(avg_lap_rank))
                avg_lap_time_millis = race_fastest_lap_data["avg_lap_time_millis"].mean()
                avg_lap_time_str = millis_to_str(avg_lap_time_millis)
            else:
                avg_lap_rank = np.nan
                avg_lap_rank_str = ""
                avg_lap_time_millis = np.nan
                avg_lap_time_str = np.nan
            source = source.append({
                "x": x,
                "constructor_name": constructor_name,
                "finish_position_str": finish_position_str,
                "finish_position_int": finish_position_int,
                "race_id": race_id,
                "year": year,
                "grid": grid,
                "grid_str": grid_str,
                "avg_lap_rank": avg_lap_rank,
                "avg_lap_rank_str": avg_lap_rank_str,
                "wdc_current_standing": current_wdc_standing,
                "wdc_current_standing_str": current_wdc_standing_str,
                "wdc_final_standing": final_standing,
                "wdc_final_standing_str": final_standing_str,
                "points": points,
                "roundNum": round_num,
                "roundName": round_name,
                "avg_lap_time_millis": avg_lap_time_millis,
                "avg_lap_time_str": avg_lap_time_str,
                "avg_finish_pos": avg_finish_pos,
                "avg_finish_pos_str": str(round(avg_finish_pos, 1))
            }, ignore_index=True)
            x += dx

    # Apply some smoothing
    source["grid_smoothed"] = source["grid"].ewm(alpha=smoothing_alpha).mean()
    source["finish_position_int_smoothed"] = source["finish_position_int"].ewm(alpha=smoothing_alpha).mean()
    source["avg_lap_rank_smoothed"] = source["avg_lap_rank"].ewm(alpha=smoothing_alpha).mean()
    source["grid"] = source["grid"].fillna("")
    source["finish_position_int"] = source["finish_position_int"].fillna(np.nan)

    source = source.sort_values(by=["year", "race_id"])
    driver_name = get_driver_name(driver_id)
    min_x = source["x"].min()
    max_x = source["x"].max()
    positions_plot = figure(
        title=u"Finishing Positions and Ranks \u2014 " + (title if title else driver_name),
        y_axis_label="Position",
        x_axis_label="Year",
        y_range=Range1d(0, 22, bounds=(0, 60)),
        x_range=Range1d(min_x, max_x, bounds=(min_x, max_x + 3)),
        tools="pan,xbox_zoom,xwheel_zoom,reset,box_zoom,wheel_zoom,save"
    )

    positions_plot.xaxis.ticker = FixedTicker(ticks=np.arange(min_year - 1, max_year + 2))
    positions_plot.yaxis.ticker = FixedTicker(ticks=np.arange(5, 61, 5).tolist() + [1])

    # Add the lines
    kwargs = {
        "x": "x",
        "source": source,
        "muted_alpha": 0.02
    }

    legend = []

    if show_mean_finish_pos:
        avg_finish_pos_line = positions_plot.line(y="avg_finish_pos", color="white", line_width=major_line_width,
                                                  line_alpha=0.7, **kwargs)
        legend.append(LegendItem(label="Yr. Avg. Finish Pos", renderers=[avg_finish_pos_line]))
        subtitle = "Year average finish position is calculated including DNFs"
        positions_plot.add_layout(Title(text=subtitle, text_font_style="italic"), "above")
    else:
        final_standing_line = positions_plot.line(y="wdc_final_standing", color="white", line_width=major_line_width,
                                                    line_alpha=0.7, **kwargs)
        wdc_finish_position_line = positions_plot.line(y="wdc_current_standing", color="green",
                                                       line_width=major_line_width,
                                                       line_alpha=0.7, **kwargs)
        legend.extend([LegendItem(label="WDC Final Year Standing", renderers=[final_standing_line]),
                       LegendItem(label="WDC Current Standing", renderers=[wdc_finish_position_line])
                       ])
    finish_position_line = positions_plot.line(y="finish_position_int", color="yellow", line_width=minor_line_width,
                                               line_alpha=0.6, **kwargs)
    grid_line = positions_plot.line(y="grid", color="orange", line_width=minor_line_width,
                                    line_alpha=0.6, **kwargs)
    avg_lap_rank_line = positions_plot.line(y="avg_lap_rank", color="hotpink", line_width=minor_line_width,
                                            line_alpha=0.7, **kwargs)
    finish_position_smoothed_line = positions_plot.line(y="finish_position_int_smoothed", color="yellow",
                                                        line_width=minor_line_width + 0.5, line_alpha=0.7,
                                                        line_dash="dashed", **kwargs)
    grid_smoothed_line = positions_plot.line(y="grid_smoothed", color="orange", line_width=minor_line_width + 0.5,
                                             line_alpha=0.7, line_dash="dashed", **kwargs)
    avg_lap_rank_smoothed_line = positions_plot.line(y="avg_lap_rank_smoothed", color="hotpink",
                                                     line_width=minor_line_width + 0.5, line_alpha=0.7,
                                                     line_dash="dashed", **kwargs)

    if smoothing_muted:
        finish_position_smoothed_line.muted = True
        grid_smoothed_line.muted = True
        avg_lap_rank_smoothed_line.muted = True
    else:
        finish_position_line.muted = True
        grid_line.muted = True
        avg_lap_rank_line.muted = True

    legend.extend([LegendItem(label="Race Finish Position", renderers=[finish_position_line]),
                   LegendItem(label="Finish Pos. Smoothed", renderers=[finish_position_smoothed_line]),
                   LegendItem(label="Avg. Lap Rank", renderers=[avg_lap_rank_line]),
                   LegendItem(label="Avg. Lap Rank Smoothed", renderers=[avg_lap_rank_smoothed_line]),
                   LegendItem(label="Grid Position", renderers=[grid_line]),
                   LegendItem(label="Grid Position Smoothed", renderers=[grid_smoothed_line])])

    if include_lap_times:
        source["avg_lap_time_scaled"] = rescale(source["avg_lap_time_millis"], positions_plot.y_range.start,
                                                positions_plot.y_range.end)
        lap_times_line = positions_plot.line(y="avg_lap_time_scaled", color="aqua", line_width=minor_line_width + 0.5,
                                             line_alpha=0.7, **kwargs)
        legend.append(LegendItem(label="Average Lap Time", renderers=[lap_times_line]))
        min_millis = source["avg_lap_time_millis"].min()
        max_millis = source["avg_lap_time_millis"].max()
        if abs(min_millis - max_millis) < 0.001:
            min_millis -= 1
        y_range = Range1d(min_millis, max_millis, bounds=(min_millis, min_millis + 60 * (max_millis - min_millis) / 22))
        positions_plot.extra_y_ranges = {"time_range": y_range}
        axis = LinearAxis(y_range_name="time_range")
        axis.formatter = DatetimeTickFormatter(**DATETIME_TICK_KWARGS)
        positions_plot.add_layout(axis, "right")

    legend = Legend(items=legend, location="top_right", glyph_height=15, spacing=2, inactive_fill_color="gray")
    positions_plot.add_layout(legend, "right")
    positions_plot.legend.click_policy = "mute"
    positions_plot.legend.label_text_font_size = "12pt"

    tooltips = [
        ("Name", driver_name),
        ("Constructor", "@constructor_name"),
        ("Year", "@year"),
        ("Round", "@roundNum - @roundName"),
        ("Grid Position", "@grid_str"),
        ("Avg. Lap Rank", "@avg_lap_rank_str"),
        ("Finishing Position", "@finish_position_str"),
        ("Points", "@points"),
        ("Final Position this year", "@wdc_final_standing_str"),
        ("Current WDC Position", "@wdc_current_standing_str"),
        ("Avg. Finish Pos. this year", "@avg_finish_pos_str")
    ]
    positions_plot.add_tools(HoverTool(show_arrow=False, tooltips=tooltips))

    # Crosshair tooltip
    positions_plot.add_tools(CrosshairTool(dimensions="both", line_color="white", line_alpha=0.6))

    # Mark team changes
    mark_team_changes(driver_years, driver_results, positions_plot, source)

    return positions_plot, source


def generate_circuit_performance_table(driver_results, driver_races, driver_id, consider_up_to=24, title=None):
    """
    Generates a table of the driver's performance at every circuit, ranked by number of wins then number of 2nd places,
    then number of 3rds, etc.
    :param driver_results: Driver results
    :param driver_races: Driver races
    :param driver_id: Driver ID
    :param consider_up_to: Place to consider up to
    :param title: If set to None, will use default, otherwise, will override default title text
    :return: Table layout, source if `return_source == True`
    """
    logging.info("Generating circuit performance table")
    if driver_results.shape[0] == 0:
        return Div(text="")
    # Calculate best circuits by first ranking by number of wins, then number of 2nd places, then 3rd, etc
    circuit_scores = defaultdict(lambda: np.zeros(consider_up_to))
    circuit_years = defaultdict(lambda: [[] for _ in range(0, consider_up_to)])
    for idx, results_row in driver_results.iterrows():
        circuit_id = driver_races.loc[results_row["raceId"], "circuitId"]
        year = driver_races.loc[results_row["raceId"], "year"]
        pos = results_row["positionOrder"]
        to_add = np.zeros(consider_up_to)
        if pos <= consider_up_to:
            to_add[pos - 1] = 1
            circuit_years[circuit_id][pos - 1].append(year)
        circuit_scores[circuit_id] += to_add
    circuit_scores = pd.DataFrame.from_dict(circuit_scores, orient="index")
    circuit_scores.index.name = "circuitId"
    circuit_scores = circuit_scores.sort_values(by=list(range(0, consider_up_to)), ascending=False)

    source = pd.DataFrame(columns=["circuit_name",
                                   "wins_int", "wins_str",
                                   "podiums_int", "podiums_str",
                                   "other_places"])
    for cid, scores_row in circuit_scores.iterrows():
        circuit_name = get_circuit_name(cid)
        wins = int(scores_row[0])
        wins_years = circuit_years[cid][0]
        wins_str = str(wins).rjust(2)
        if int(wins) > 0:
            wins_str += " (" + rounds_to_str(wins_years) + ")"
        podiums = int(scores_row[0] + scores_row[1] + scores_row[2])
        podiums_years = wins_years + circuit_years[cid][1] + circuit_years[cid][2]
        podiums_str = str(podiums).rjust(2)
        if int(podiums) > 0:
            podiums_str += " (" + rounds_to_str(podiums_years) + ")"
        other_places = ""
        place = 4
        for num_places in scores_row.values[3:]:
            num_places = int(num_places)
            if num_places > 0:
                ordinal = int_to_ordinal(place)
                other_places += str(num_places) + " " + ordinal + ("s" if num_places > 1 else "") + ", "
            place += 1
        other_places = other_places[:-2] if len(other_places) > 0 else other_places
        source = source.append({
            "circuit_name": circuit_name,
            "wins_int": wins,
            "wins_str": wins_str,
            "podiums_int": podiums,
            "podiums_str": podiums_str,
            "other_places": other_places
        }, ignore_index=True)

    best_circuits_columns = [
        TableColumn(field="circuit_name", title="Circuit Name", width=150),
        TableColumn(field="wins_str", title="Wins", width=160),
        TableColumn(field="podiums_str", title="Podiums", width=200),
        TableColumn(field="other_places", title="Other Finishes", width=260),
    ]

    circuits_table = DataTable(source=ColumnDataSource(data=source), columns=best_circuits_columns,
                               index_position=None, min_height=530)
    driver_name = get_driver_name(driver_id, include_flag=False)
    if title:
        title = Div(text=title)
    else:
        title = Div(text=f"<h2><b>What were {driver_name}'s Best Circuits?</b></h2>")

    layout = column([title, row([circuits_table], sizing_mode="stretch_width")], sizing_mode="stretch_width")
    return layout


def generate_finishing_position_bar_plot(driver_results, consider_up_to=24, plot_height=400):
    """
    Bar plot showing distribution of race finishing positions.
    :param driver_results: Driver results
    :param consider_up_to: Place to consider up to
    :param plot_height: Plot height
    :return: Finishing position distribution plot layout
    """
    logging.info("Generating finishing position bar plot")
    results = driver_results["positionText"].apply(position_text_to_str).apply(int_to_ordinal)
    n = results.shape[0]
    if n == 0:
        return Div(text="")
    results = results.value_counts()
    names = [int_to_ordinal(i) for i in range(1, consider_up_to + 1)] + ["RET"]
    bar_dict = OrderedDict({k: 0 for k in names})
    for name, count in results.items():
        if name in bar_dict:
            bar_dict[name] = count
    names = []
    counts = []
    percents = []
    for k, v in bar_dict.items():
        names.append(k)
        counts.append(v)
        percents.append(str(round(100 * v / n, 1)) + "%")

    source = ColumnDataSource({
        "name": names,
        "count": counts,
        "percent": percents
    })
    finish_position_plot = figure(x_range=names,
                                  title=u"Race Finishing Positions \u2014 Doesn't include DNQ, NC, and DSQ finishes",
                                  tools="hover",
                                  tooltips="@name: @count (@percent)",
                                  plot_height=plot_height)
    finish_position_plot.y_range.start = 0
    finish_position_plot.y_range.end = max(counts) + 5
    finish_position_plot.vbar(x="name", top="count", width=0.9, source=source, color="yellow", fill_alpha=0.8,
                              line_alpha=0.9)

    # Add the other y axis
    max_y = finish_position_plot.y_range.end
    y_range = Range1d(start=0, end=max_y / n)
    finish_position_plot.extra_y_ranges = {"percent_range": y_range}
    axis = LinearAxis(y_range_name="percent_range")
    axis.formatter = NumeralTickFormatter(format="0.0%")
    finish_position_plot.add_layout(axis, "right")

    return finish_position_plot


def generate_wdc_position_bar_plot(positions_source, consider_up_to=24, plot_height=400):
    """
    Bar plot showing distribution of WDC finishing positions.
    :param positions_source: Position source
    :param consider_up_to: Position to plot up to
    :param plot_height: Plot height
    :return: WDC position distribution plot layout
    """
    logging.info("Generating WDC finishing position bar plot")
    if isinstance(positions_source, dict):
        return Div(text="")
    results = positions_source.set_index("year")
    results = results.loc[~results.index.duplicated(keep="last")]
    positions = results.copy()
    n = results.shape[0]
    if n == 0:
        return Div(text="")
    results = results["wdc_final_standing"].apply(int_to_ordinal).value_counts()
    names = [int_to_ordinal(i) for i in range(1, consider_up_to + 1)]
    bar_dict = OrderedDict({k: 0 for k in names})
    for name, count in results.items():
        name = str(name)
        if name in bar_dict:
            bar_dict[name] = count
    names = []
    counts = []
    percents = []
    years = []
    for k, v in bar_dict.items():
        names.append(k)
        counts.append(v)
        if n > 0:
            percents.append(str(round(100 * v / n, 1)) + "%")
        else:
            percents.append("0.0%")
        years_this_pos = positions[positions["wdc_final_standing"].fillna(-1).apply(int_to_ordinal) ==
                                   int_to_ordinal(k)]
        years_str = ", ".join(years_this_pos.index.values.astype(str).tolist())
        if len(years_str) > 0:
            years_str = "(" + years_str + ")"
        years.append(years_str)

    source = ColumnDataSource({
        "name": names,
        "count": counts,
        "percent": percents,
        "years": years
    })
    wdc_finish_position_plot = figure(x_range=names,
                                      title=u"WDC Finishing Positions",
                                      tools="hover",
                                      tooltips="@name: @count @years",
                                      plot_height=plot_height)
    wdc_finish_position_plot.yaxis.ticker = FixedTicker(ticks=np.arange(0, 500))
    wdc_finish_position_plot.y_range.start = 0
    wdc_finish_position_plot.y_range.end = max(counts) + 1
    wdc_finish_position_plot.vbar(x="name", top="count", width=0.9, source=source, color="white", fill_alpha=0.8,
                                  line_alpha=0.9)

    # Add the other y axis
    max_y = wdc_finish_position_plot.y_range.end
    y_range = Range1d(start=0, end=max_y / n)
    wdc_finish_position_plot.extra_y_ranges = {"percent_range": y_range}
    axis = LinearAxis(y_range_name="percent_range")
    axis.formatter = NumeralTickFormatter(format="0.0%")
    wdc_finish_position_plot.add_layout(axis, "right")

    return wdc_finish_position_plot


def mark_team_changes(driver_years, driver_results, fig, source):
    """
    Mark team changes with a vertical line.
    :param driver_years: Driver years
    :param driver_results: Driver results
    :param fig: Figure
    :param source: Source (must have "x" column)
    :return: None
    """
    logging.info("Marking team changes")
    if driver_years.shape[0] == 0:
        return
    min_year = driver_years.min()
    max_year = driver_years.max()
    prev_constructor = -1
    color_gen = ColorDashGenerator()
    label_kwargs = dict(render_mode="canvas",
                        text_color="white",
                        text_font_size="12pt",
                        angle=math.pi / 4)
    for year in range(min_year, max_year + 1):
        year_races = races[races["year"] == year]
        year_driver_results = driver_results[driver_results["raceId"].isin(year_races.index)]
        if year_driver_results.shape[0] == 0:
            continue
        for race_id in year_races.sort_index().index:
            race_results = year_driver_results[year_driver_results["raceId"] == race_id]
            if race_results.shape[0] > 0:
                curr_constructor = race_results["constructorId"].values[0]
                if curr_constructor != prev_constructor:  # Mark the constructor change
                    x = source[source["race_id"] == race_id]["x"]
                    if x.shape[0] > 0:
                        x = x.values[0]
                    else:
                        continue
                    color, _ = color_gen.get_color_dash(None, curr_constructor)
                    line = Span(line_color=color, location=x, dimension="height", line_alpha=0.4, line_width=3.2)
                    fig.add_layout(line)
                    label = Label(x=x + 0.01, y=16, text=get_constructor_name(curr_constructor), **label_kwargs)
                    fig.add_layout(label)
                    prev_constructor = curr_constructor


def generate_team_performance_layout(driver_races, positions_source, driver_results):
    """
    Generates a table showing the following during whole career and per-constructor:
    - Dates at that constructor
    - Highest WDC finish
    - Num races
    - Num poles and % pole
    - Num wins and win %
    - Num podiums and podium %
    - Num finishes and finish %
    - Num mechanical-related DNFs and mechanical DNF %
    - Num crash-related DNFs and crash DNF %
    - Points scored and points per race
    :param driver_races: Driver races
    :param positions_source: Positions source
    :param driver_results: Driver results
    :return: Team performance layout
    """
    logging.info("Generating team performance table")
    if isinstance(positions_source, dict):
        return Div(text=""), pd.DataFrame()
    positions_source = positions_source.set_index("race_id")
    source = pd.DataFrame(columns=["constructor_name", "years", "wdc_final_standings", "num_races", "num_poles",
                                   "num_wins", "num_podiums", "num_finishes", "num_mechanical_dnf", "num_crash_dnf",
                                   "total_points"]).set_index("constructor_name")
    for race_id in driver_races.index:
        if race_id in positions_source.index:
            race_results = driver_results[driver_results["raceId"] == race_id].iloc[0]
            positions_row = positions_source.loc[race_id]
            constructor_name = positions_row["constructor_name"]
            year = int(positions_row["year"])
            wdc_final_standing = positions_row["wdc_final_standing"]
            finish_pos = positions_row["finish_position_int"]
            win = 1 if finish_pos == 1 else 0
            podium = 1 if 3 >= finish_pos >= 1 else 0
            pole = 1 if race_results["grid"] == 1 else 0
            points = race_results["points"]
            status = race_results["statusId"]
            status = get_status_classification(status)
            finished = 1 if status == "finished" else 0
            mechanical_dnf = 1 if status == "mechanical" else 0
            crash_dnf = 1 if status == "crash" else 0
            if constructor_name in source.index:
                source_row = source.loc[constructor_name]
                years = source_row["years"]
                wdc_final_standings = source_row["wdc_final_standings"]
                if year not in years:
                    years.append(year)
                    wdc_final_standings.append(wdc_final_standing)
                source.loc[constructor_name] = [
                    years,
                    wdc_final_standings,
                    source_row["num_races"] + 1,
                    source_row["num_poles"] + pole,
                    source_row["num_wins"] + win,
                    source_row["num_podiums"] + podium,
                    source_row["num_finishes"] + finished,
                    source_row["num_mechanical_dnf"] + mechanical_dnf,
                    source_row["num_crash_dnf"] + crash_dnf,
                    source_row["total_points"] + points
                ]
            else:
                source.loc[constructor_name] = {
                    "years": [year],
                    "wdc_final_standings": [wdc_final_standing],
                    "num_races": 1,
                    "num_poles": pole,
                    "num_wins": win,
                    "num_podiums": podium,
                    "num_finishes": finished,
                    "num_mechanical_dnf": mechanical_dnf,
                    "num_crash_dnf": crash_dnf,
                    "total_points": points
                }

    # Add the all career column
    years = []
    wdc_final_standings = []
    for idx, source_row in source.iterrows():
        years.extend(source_row["years"])
        wdc_final_standings.extend(source_row["wdc_final_standings"])
    source.loc["Career Total"] = {
        "years": years,
        "wdc_final_standings": wdc_final_standings,
        "num_races": source["num_races"].sum(),
        "num_poles": source["num_poles"].sum(),
        "num_wins": source["num_wins"].sum(),
        "num_podiums": source["num_podiums"].sum(),
        "num_finishes": source["num_finishes"].sum(),
        "num_mechanical_dnf": source["num_mechanical_dnf"].sum(),
        "num_crash_dnf": source["num_crash_dnf"].sum(),
        "total_points": source["total_points"].sum()
    }

    source["total_points"] = source["total_points"].apply(lambda p: p if abs(int(p) - p) > 0.01 else int(p))
    source["years_str"] = source["years"].apply(lambda x: rounds_to_str(x, 1000))
    source["dnfs"] = source["num_races"] - source["num_finishes"]
    source["wdc_final_standings_str"] = None
    for idx, source_row in source.iterrows():
        wdc_final_standings_str = ""
        standings_dict = {y: s for y, s in zip(source_row["years"], source_row["wdc_final_standings"])}
        standings_dict = {k: v for k, v in sorted(standings_dict.items(), key=lambda item: item[1])}
        for i, k in enumerate(standings_dict):
            if i > 3 and idx == "Career Total":
                wdc_final_standings_str = wdc_final_standings_str[:-2] + "...  "
                break
            wdc_final_standings_str += int_to_ordinal(standings_dict[k]) + f" ({k}), "
        wdc_final_standings_str = wdc_final_standings_str if len(wdc_final_standings_str) == 0 else \
            wdc_final_standings_str[:-2]
        source.loc[idx, "wdc_final_standings_str"] = wdc_final_standings_str

    # Add percents
    def add_pct(source_row, column):
        if source_row["num_races"] == 0:
            pct = 0
        else:
            pct = 100 * source_row[column] / source_row["num_races"]
        return str(source_row[column]) + f" ({str(round(pct, 1))}%)"

    def add_ppr(source_row, column):
        if source_row["num_races"] == 0:
            pct = 0
        else:
            pct = source_row[column] / source_row["num_races"]
        return str(source_row[column]) + f" ({str(round(pct, 1))} pts/race)"
    source["num_poles_str"] = source.apply(lambda r: add_pct(r, "num_poles"), axis=1)
    source["num_wins_str"] = source.apply(lambda r: add_pct(r, "num_wins"), axis=1)
    source["num_podiums_str"] = source.apply(lambda r: add_pct(r, "num_podiums"), axis=1)
    source["num_finishes_str"] = source.apply(lambda r: add_pct(r, "num_finishes"), axis=1)
    source["num_mechanical_dnf_str"] = source.apply(lambda r: add_pct(r, "num_mechanical_dnf"), axis=1)
    source["num_crash_dnf_str"] = source.apply(lambda r: add_pct(r, "num_crash_dnf"), axis=1)
    source["total_points_str"] = source.apply(lambda r: add_ppr(r, "total_points"), axis=1)

    team_performance_columns = [
        TableColumn(field="constructor_name", title="Constructor Name", width=150),
        TableColumn(field="years_str", title="Years", width=100),
        TableColumn(field="wdc_final_standings_str", title="WDC Finishes", width=250),
        TableColumn(field="num_races", title="Races", width=50),
        TableColumn(field="num_poles_str", title="Poles", width=75),
        TableColumn(field="num_wins_str", title="Wins", width=75),
        TableColumn(field="num_podiums_str", title="Podiums", width=75),
        TableColumn(field="num_finishes_str", title="Finishes", width=75),
        TableColumn(field="num_mechanical_dnf_str", title="Mech. DNFs", width=75),
        TableColumn(field="num_crash_dnf_str", title="Crash DNFs", width=75),
        TableColumn(field="total_points_str", title="Total Points", width=100),
    ]

    team_performance_columns_table = DataTable(source=ColumnDataSource(data=source), columns=team_performance_columns,
                                               index_position=None, height=source.shape[0] * 40)
    title = Div(text=f"<h2><b>Performance at each Constructor</b></h2>")

    # Generate bar plot
    bar_plot_bars = ["Races", "Points", "Poles", "Wins", "Podiums", "DNFs"]
    counts = source[["num_races", "total_points", "num_poles", "num_wins", "num_podiums", "dnfs"]].values.flatten()
    counts = counts.tolist()
    x = [(name, data_name) for name in source.index for data_name in bar_plot_bars]
    performance_plot = figure(x_range=FactorRange(*x),
                              plot_height=350,
                              title="Performance by Constructor (note the point system may have changed over time)",
                              tools="hover",
                              tooltips="@x: @count")
    plot_source = ColumnDataSource(dict(x=x, count=counts))
    performance_plot.vbar(x="x", top="count", width=0.9, source=plot_source, line_color="white",
                          fill_color=factor_cmap("x", palette=palette, factors=bar_plot_bars, start=1, end=2))
    performance_plot.y_range.start = 0
    performance_plot.x_range.range_padding = 0.1
    performance_plot.xaxis.major_label_orientation = 1
    performance_plot.xgrid.grid_line_color = None
    performance_plot.xaxis.group_text_color = "white"
    performance_plot.xaxis.group_text_font_size = "11pt"

    return column([title, performance_plot, row([team_performance_columns_table], sizing_mode="stretch_width")],
                  sizing_mode="stretch_width"), source


def generate_win_plot(positions_source, driver_id=None, constructor_id=None):
    """
    Plots number of races, win percentage, number of wins, podium percentage, and number of podiums on the same plot
    (2 different axes on each side).
    :param positions_source: Positions source
    :param driver_id: Driver ID, if left None then the "subtitle" won't be included
    :param constructor_id: Constructor ID, must be set to use constructor mode
    :return: Plot layout
    """
    # TODO refactor to add support for top-n finishes too and dynamically come up with n (like top 6),
    #  see circuitdriver.generate_win_plot and Trello
    # TODO refactor to add support for points and points per race percent (not really needed cuz points systems change?)
    logging.info("Generating win plot")
    if isinstance(positions_source, dict):
        return Div(text="")
    win_source = pd.DataFrame(columns=["x", "n_races",
                                       "win_pct", "wins", "win_pct_str",
                                       "podium_pct", "podiums", "podium_pct_str",
                                       "dnf_pct", "dnfs", "dnf_pct_str",
                                       "constructor_name", "year", "wdc_final_standing", "wdc_final_standing_str",
                                       "roundNum", "roundName"])
    wins = 0
    podiums = 0
    dnfs = 0
    n_races = 0
    for idx, row in positions_source.sort_values(by="x").iterrows():
        x = row["x"]
        if constructor_id:
            poses = row["finish_positions"]
            for pos in poses:
                if not np.isnan(pos):
                    wins += 1 if pos == 1 else 0
                    podiums += 1 if 3 >= pos > 0 else 0
            dnfs += row["num_dnfs_this_race"]
        else:
            pos = row["finish_position_int"]
            if not np.isnan(pos):
                wins += 1 if pos == 1 else 0
                podiums += 1 if 3 >= pos > 0 else 0
        n_races += 1
        win_pct = wins / n_races
        podium_pct = podiums / n_races
        dnf_pct = dnfs / n_races
        championship_final_standing = row["wdc_final_standing" if constructor_id is None else "wcc_final_standing"]
        win_source = win_source.append({
            "x": x,
            "n_races": n_races,
            "wins": wins,
            "win_pct": win_pct,
            "podiums": podiums,
            "podium_pct": podium_pct,
            "dnfs": dnfs,
            "dnf_pct": dnf_pct,
            "dnf_pct_str": str(round(100 * dnf_pct, 1)) + "%",
            "constructor_name": row["constructor_name"],
            "year": row["year"],
            "wdc_final_standing": championship_final_standing,
            "wdc_final_standing_str": int_to_ordinal(championship_final_standing),
            "roundNum": row["roundNum"],
            "roundName": row["roundName"],
            "win_pct_str": str(round(100 * win_pct, 1)) + "%",
            "podium_pct_str": str(round(100 * podium_pct, 1)) + "%"
        }, ignore_index=True)

    min_x = positions_source["x"].min()
    max_x = positions_source["x"].max()
    title = "Wins and Podiums"
    if driver_id:
        driver_name = get_driver_name(driver_id)
        title += u"\u2014 " + driver_name
    elif constructor_id:
        constructor_name = get_constructor_name(constructor_id)
        title += u"\u2014 " + constructor_name
    max_podium = win_source["podiums"].max()
    max_dnfs = win_source["dnfs"].max()
    win_plot = figure(
        title=title,
        y_axis_label="",
        x_axis_label="Year",
        x_range=Range1d(min_x, max_x, bounds=(min_x, max_x + 3)),
        tools="pan,xbox_zoom,reset,box_zoom,wheel_zoom,save",
        y_range=Range1d(0, max(max_podium, max_dnfs), bounds=(0, 1000))
    )
    if constructor_id:
        subtitle = "Win and podium percent is calculated as num. wins / num. races entered, and thus podium pct. may " \
                   "theoretically exceed 100%"
        win_plot.add_layout(Title(text=subtitle, text_font_style="italic"), "above")
    max_podium_pct = win_source["podium_pct"].max()
    max_dnf_pct = win_source["dnf_pct"].max()
    if max_podium == 0:
        return Div()  # Theoretically this case could be handled but not really useful,
        # will have to be handled when doing the top-n calculate
    if max_podium > max_dnfs:
        k = max_podium / max_podium_pct
    elif max_dnf_pct > 0:
        k = max_dnfs / max_dnf_pct
    else:
        k = 1
    win_source["podium_pct_scaled"] = k * win_source["podium_pct"]
    win_source["win_pct_scaled"] = k * win_source["win_pct"]
    win_source["dnf_pct_scaled"] = k * win_source["dnf_pct"]

    # Other y axis
    y_range = Range1d(start=0, end=max(max_podium_pct, max_dnf_pct), bounds=(-0.02, 1000))
    win_plot.extra_y_ranges = {"percent_range": y_range}
    axis = LinearAxis(y_range_name="percent_range")
    axis.formatter = NumeralTickFormatter(format="0.0%")
    win_plot.add_layout(axis, "right")

    kwargs = {
        "x": "x",
        "line_width": 2,
        "line_alpha": 0.7,
        "source": win_source,
        "muted_alpha": 0.02
    }
    races_line = win_plot.line(y="n_races", color="white", **kwargs)
    wins_line = win_plot.line(y="wins", color="green", **kwargs)
    win_pct_line = win_plot.line(y="win_pct_scaled", color="green", line_dash="dashed", **kwargs)
    podiums_line = win_plot.line(y="podiums", color="yellow", **kwargs)
    podium_pct_line = win_plot.line(y="podium_pct_scaled", color="yellow", line_dash="dashed", **kwargs)

    legend = [LegendItem(label="Number of Races", renderers=[races_line]),
              LegendItem(label="Number of Wins", renderers=[wins_line]),
              LegendItem(label="Win Percentage", renderers=[win_pct_line]),
              LegendItem(label="Number of Podiums", renderers=[podiums_line]),
              LegendItem(label="Podium Percentage", renderers=[podium_pct_line])]

    if constructor_id:
        dnfs_line = win_plot.line(y="dnfs", color="aqua", **kwargs)
        dnf_pct_line = win_plot.line(y="dnf_pct_scaled", color="aqua", line_dash="dashed", **kwargs)
        legend.extend([
            LegendItem(label="Number of DNFs", renderers=[dnfs_line]),
            LegendItem(label="DNF Percentage", renderers=[dnf_pct_line])
        ])

    legend = Legend(items=legend, location="top_right", glyph_height=15, spacing=2, inactive_fill_color="gray")
    win_plot.add_layout(legend, "right")
    win_plot.legend.click_policy = "mute"
    win_plot.legend.label_text_font_size = "12pt"

    # Hover tooltip
    tooltips = [
        ("Number of Races", "@n_races"),
        ("Number of Wins", "@wins (@win_pct_str)"),
        ("Number of Podiums", "@podiums (@podium_pct_str)"),
        ("Constructor", "@constructor_name"),
        ("Year", "@year"),
        ("Round", "@roundNum - @roundName"),
        ("Final Position this year", "@wdc_final_standing_str")
    ]
    if constructor_id:
        tooltips.append(("Number of DNFs", "@dnfs (@dnf_pct_str)"))
    win_plot.add_tools(HoverTool(show_arrow=False, tooltips=tooltips))

    # Crosshair tooltip
    win_plot.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

    return win_plot


def generate_spvfp_scatter(driver_results, driver_races, driver_driver_standings, include_year_labels=False,
                           include_race_labels=False, include_constructor_name=False, include_driver_name=True,
                           color_drivers=False):
    """
    Plot a scatter of quali position vs finish position and draw the y=x line
    :param driver_results: Driver results
    :param driver_races: Driver races
    :param driver_driver_standings: Driver driver standings
    :param include_year_labels: Whether to include year labels
    :param include_race_labels: Whether to include race flag as labels
    :param include_constructor_name: Whether to include the constructor name in the tooltip
    :param include_driver_name: Whether to include the driver name in the tooltip
    :param color_drivers: Whether to color based on which driver it is
    :return: Start pos. vs finish pos. scatter layout
    """
    logging.info("Generating start pos. vs finish pos. scatter")
    spvfp_scatter = figure(title=u"Starting Position vs Finish Position \u2014 Saturday vs Sunday Performance",
                           x_axis_label="Grid Position",
                           y_axis_label="Finishing Position (Official Classification)",
                           x_range=Range1d(0, 22, bounds=(0, 60)),
                           y_range=Range1d(0, 22, bounds=(0, 60)),
                           tools="pan,wheel_zoom,reset,save")
    spvfp_scatter.xaxis.ticker = FixedTicker(ticks=np.arange(5, 61, 5).tolist() + [1])
    spvfp_scatter.yaxis.ticker = FixedTicker(ticks=np.arange(5, 61, 5).tolist() + [1])

    source = pd.DataFrame(columns=["sp", "sp_str",
                                   "fp", "fp_str",
                                   "marker", "color", "size",
                                   "year", "constructor_name", "driver_name",
                                   "roundNum", "roundName", "roundFlag"])
    color_gen = ColorDashGenerator()
    for idx, row in driver_results.iterrows():
        sp = row["grid"]
        sp_str = int_to_ordinal(sp)
        fp = row["positionOrder"]
        constructor_name = get_constructor_name(row["constructorId"])
        did = row["driverId"]
        driver_name = get_driver_name(did)
        status_id = row["statusId"]
        year = driver_races.loc[row["raceId"], "year"]
        classification = get_status_classification(status_id)
        fp_str, _ = result_to_str(fp, status_id)
        if classification == "mechanical" or classification == "crash":
            marker = "x"
            color = "red" if classification == "crash" else "orange"
            size = 10
        else:
            marker = "circle"
            if color_drivers:
                color = color_gen.get_color_dash(did, did)[0]
            else:
                color = "white"
            size = 8
        current_standing = driver_driver_standings[driver_driver_standings["raceId"] == row["raceId"]]
        if current_standing.shape[0] and current_standing.shape[0] > 0:
            round_num = current_standing["roundNum"].values[0]
            round_name = current_standing["roundName"].values[0]
        else:
            round_num = ""
            round_name = ""

        source = source.append({
            "sp": sp,
            "fp": fp,
            "marker": marker,
            "color": color,
            "year": year,
            "sp_str": sp_str,
            "fp_str": fp_str,
            "size": size,
            "constructor_name": constructor_name,
            "driver_name": driver_name,
            "roundNum": round_num,
            "roundName": round_name,
            "roundFlag": round_name[:2]
        }, ignore_index=True)

    spvfp_scatter.scatter(x="sp", y="fp", source=source, size="size", color="color", marker="marker")
    spvfp_scatter.line(x=[-60, 60], y=[-60, 60], color="white", line_alpha=0.5)

    # Labels
    marker_label_kwargs = dict(x="sp",
                               y="fp",
                               level="glyph",
                               x_offset=1.1,
                               y_offset=1.1,
                               source=ColumnDataSource(source),
                               render_mode="canvas",
                               text_color="white",
                               text_font_size="10pt")
    if include_year_labels:
        labels = LabelSet(text="year", **marker_label_kwargs)
        spvfp_scatter.add_layout(labels)
    if include_race_labels:
        labels = LabelSet(text="roundFlag", **marker_label_kwargs)
        spvfp_scatter.add_layout(labels)

    text_label_kwargs = dict(render_mode="canvas",
                             text_color="white",
                             text_font_size="12pt",
                             border_line_color="white",
                             border_line_alpha=0.7)
    label1 = Label(x=1, y=21, text=" Finish lower than started ", **text_label_kwargs)
    label2 = Label(x=15, y=0, text=" Finish higher than start ", **text_label_kwargs)
    spvfp_scatter.add_layout(label1)
    spvfp_scatter.add_layout(label2)

    # Hover tooltip
    tooltips = [
        ("Starting Position", "@sp_str"),
        ("Finishing Position", "@fp_str"),
        ("Round", "@roundNum - @roundName"),
        ("Year", "@year")
    ]
    if include_constructor_name:
        tooltips.append(("Constructor Name", "@constructor_name"))
    if include_driver_name:
        tooltips.insert(0, ("Driver Name", "@driver_name"))
    spvfp_scatter.add_tools(HoverTool(show_arrow=False, tooltips=tooltips))

    # Crosshair
    spvfp_scatter.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

    return spvfp_scatter


def generate_mltr_fp_scatter(driver_results, driver_races, driver_driver_standings,
                             include_year_labels=False, include_race_labels=False, include_constructor_name=False,
                             include_driver_name=True, color_drivers=False):
    """
    Plot scatter of mean lap time rank (x) vs finish position (y) to get a sense of what years the driver out-drove the
    car.
    :param driver_results: Driver results
    :param driver_races: Driver races
    :param driver_driver_standings: Driver driver standings
    :param include_year_labels: Whether to include the year labels
    :param include_race_labels: Whether to include race flag as labels
    :param include_constructor_name: Whether to include the constructor name in the tooltip
    :param include_driver_name: Whether to include the driver name in the tooltip
    :param color_drivers: Whether to color based on which driver it is
    :return: Mean lap time rank vs finish position scatter plot layout
    """
    # TODO change this method to use mean lap time percent, where percent is compared to fastest mean lap time
    logging.info("Generating mean lap time rank vs finish pos scatter plot")
    source = pd.DataFrame(columns=["year", "constructor_name", "driver_name",
                                   "fp", "fp_str",
                                   "mltr", "mltr_str", "mlt_str",
                                   "marker", "color", "size",
                                   "roundNum", "roundName", "roundFlag"])
    color_gen = ColorDashGenerator()
    for idx, row in driver_results.iterrows():
        rid = row["raceId"]
        year = driver_races.loc[rid, "year"]
        constructor_name = get_constructor_name(row["constructorId"])
        did = row["driverId"]
        driver_name = get_driver_name(did)
        fp = row["positionOrder"]
        status_id = row["statusId"]
        classification = get_status_classification(status_id)
        fp_str, _ = result_to_str(fp, status_id)
        if classification == "mechanical" or classification == "crash":
            marker = "x"
            color = "red" if classification == "crash" else "orange"
            size = 10
        else:
            marker = "circle"
            if color_drivers:
                color = color_gen.get_color_dash(did, did)[0]
            else:
                color = "white"
            size = 8
        year_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"] == rid]
        year_fastest_lap_data = year_fastest_lap_data.set_index("driver_id")
        avg_lap_ranks = year_fastest_lap_data["avg_lap_time_millis"].rank()
        mltr_str = "DNF"
        if did in avg_lap_ranks.index:
            mltr = avg_lap_ranks.loc[did]
            avg_lap_time_str = year_fastest_lap_data.loc[did, "avg_lap_time_str"]
            if isinstance(avg_lap_time_str, pd.Series):
                avg_lap_time_str = avg_lap_time_str.values[0]
            if isinstance(mltr, pd.Series):
                mltr = mltr.values[0]
            if not np.isnan(mltr):
                mltr = int(mltr)
                mltr_str = int_to_ordinal(mltr)
        else:
            avg_lap_time_str = ""
            mltr = np.nan
        if np.isnan(mltr):
            mltr = fp
        current_standing = driver_driver_standings[driver_driver_standings["raceId"] == row["raceId"]]
        if current_standing.shape[0] and current_standing.shape[0] > 0:
            round_num = current_standing["roundNum"].values[0]
            round_name = current_standing["roundName"].values[0]
        else:
            round_num = ""
            round_name = ""
        source = source.append({
            "year": year,
            "fp": fp,
            "mltr": mltr,
            "mltr_str": mltr_str,
            "mlt_str": avg_lap_time_str,
            "marker": marker,
            "color": color,
            "size": size,
            "fp_str": fp_str,
            "constructor_name": constructor_name,
            "driver_name": driver_name,
            "roundNum": round_num,
            "roundName": round_name,
            "roundFlag": round_name[:2]
        }, ignore_index=True)

    mltr_fp_scatter = figure(title=u"Average Lap Time Rank vs Finish Position \u2014 "
                                   u"When did they out-driver their car?",
                             x_axis_label="Mean Lap Time Rank",
                             y_axis_label="Finishing Position (Official Classification)",
                             x_range=Range1d(0, 22, bounds=(0, 60)),
                             y_range=Range1d(0, 22, bounds=(0, 60)),
                             tools="pan,wheel_zoom,reset,save")
    mltr_fp_scatter.xaxis.ticker = FixedTicker(ticks=np.arange(5, 61, 5).tolist() + [1])
    mltr_fp_scatter.yaxis.ticker = FixedTicker(ticks=np.arange(5, 61, 5).tolist() + [1])

    mltr_fp_scatter.scatter(x="mltr", y="fp", source=source, size="size", color="color", marker="marker")
    mltr_fp_scatter.line(x=[-60, 60], y=[-60, 60], color="white", line_alpha=0.5)

    # Labels
    marker_label_kwargs = dict(x="mltr",
                               y="fp",
                               level="glyph",
                               x_offset=1.1,
                               y_offset=1.1,
                               source=ColumnDataSource(source),
                               render_mode="canvas",
                               text_color="white",
                               text_font_size="10pt")
    if include_year_labels:
        print("here")
        labels = LabelSet(text="year", **marker_label_kwargs)
        mltr_fp_scatter.add_layout(labels)
    if include_race_labels:
        labels = LabelSet(text="roundFlag", **marker_label_kwargs)
        mltr_fp_scatter.add_layout(labels)

    text_label_kwargs = dict(render_mode="canvas",
                             text_color="white",
                             text_font_size="12pt",
                             border_line_color="white",
                             border_line_alpha=0.7)
    label1 = Label(x=1, y=21, text=" Finish lower than expected ", **text_label_kwargs)
    label2 = Label(x=13, y=0, text=" Finish higher than expected ", **text_label_kwargs)
    mltr_fp_scatter.add_layout(label1)
    mltr_fp_scatter.add_layout(label2)

    # Hover tooltip
    tooltips = [
        ("Year", "@year"),
        ("Avg. Lap Time", "@mlt_str"),
        ("Avg. Lap Time Rank", "@mltr_str"),
        ("Finishing Position", "@fp_str"),
        ("Round", "@roundNum - @roundName")
    ]
    if include_constructor_name:
        tooltips.append(("Constructor Name", "@constructor_name"))
    if include_driver_name:
        tooltips.insert(0, ("Driver Name", "@driver_name"))
    mltr_fp_scatter.add_tools(HoverTool(show_arrow=False, tooltips=tooltips))

    # Crosshair
    mltr_fp_scatter.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

    return mltr_fp_scatter


def generate_teammate_comparison_line_plot(positions_source, constructor_results, driver_years, driver_results,
                                           driver_id):
    """
    Teammate comparison line plot
    :param positions_source: Positions source
    :param constructor_results: Constructor results (all results from the constructor this driver was at)
    :param driver_years: Driver years
    :param driver_results: Driver results
    :param driver_id: Driver ID
    :return: Plot layout
    """
    slider, plot, source = driverconstructor.generate_teammate_comparison_line_plot(positions_source,
                                                                                    constructor_results, driver_id,
                                                                                    return_components_and_source=True)
    mark_team_changes(driver_years, driver_results, plot, source)
    return column([slider, plot], sizing_mode="stretch_width")


def generate_stats_layout(driver_years, driver_races, performance_source, driver_results, driver_id):
    """
    Includes some information not found in the "Career Total" category of the team performance table.
    :param driver_years: Driver years
    :param driver_races: Driver races
    :param performance_source: Performance source
    :param driver_results: Driver results
    :param driver_id: Driver ID
    :return: Stats layout
    """
    logging.info("Generating driver stats layout")
    if performance_source.shape[0] <= 1:
        return Div(text="")
    driver_name = get_driver_name(driver_id, include_flag=False)
    driver_entry = drivers.loc[driver_id]
    nationality = driver_entry["nationality"].lower().title()
    nationality = nationality_to_flag(nationality.lower()) + nationality
    dob = driver_entry["dob"]
    dob = datetime.strptime(dob, "%Y-%M-%d").strftime("%d %B %Y").strip("0")

    num = driver_entry["number"]
    years_active = rounds_to_str(driver_years)
    teams = ", ".join(performance_source.index.unique().tolist()[:-1])

    first_year = np.min(driver_years)
    first_rid = driver_races[driver_races["year"] == first_year].sort_values(by="round").index.values[0]
    first_race_name = str(first_year) + " " + get_race_name(first_rid)
    last_year = np.max(driver_years)
    last_rid = driver_races[driver_races["year"] == last_year].sort_values(by="round").index.values[-1]
    last_race_name = str(last_year) + " " + get_race_name(last_rid)

    wins = []
    driver_results = driver_results.set_index("raceId")
    last_win_rid = -1
    last_win_year = -1
    for rid, race_row in driver_races.iterrows():
        position = driver_results.loc[rid, "positionOrder"]
        if isinstance(position, Series):
            position = position.values[0]
        if rid in driver_results.index and position == 1:
            year = race_row["year"]
            if year > last_win_year and rid > last_win_rid:
                last_win_rid = rid
                last_win_year = year
            wins.append(rid)
    if len(wins) > 0:
        first_win_rid = wins[0]
        first_win_year = races.loc[first_win_rid, "year"]
        first_win_name = str(first_win_year) + " " + get_race_name(first_win_rid)
        last_win_name = get_race_name(last_win_rid, include_year=True)

    performance_source = performance_source.loc["Career Total"]
    num_podiums_str = performance_source["num_podiums_str"]
    num_wins_str = performance_source["num_wins_str"]
    career_points = performance_source["total_points"]
    num_races = performance_source["num_races"]
    mean_sp = round(driver_results["grid"].mean(), 1)
    mean_fp = round(driver_results["positionOrder"].mean(), 1)

    num_championships = 0
    championships_str = " ("
    for year, standing in zip(performance_source["years"], performance_source["wdc_final_standings"]):
        if standing == 1:
            num_championships += 1
            championships_str += str(year) + ", "
    if num_championships == 0:
        championships_str = str(np.max(performance_source["wdc_final_standings"]))
    else:
        championships_str = str(num_championships) + championships_str[:-2] + ")"

    # Construct the HTML
    header_template = """
    <h2 style="text-align: center;"><b>{}</b></h2>
    """

    template = """
    <pre><b>{}</b> {}<br></pre>
    """

    driver_stats = header_template.format("Driver Stats")
    driver_stats += template.format("Name: ".ljust(20), driver_name)
    driver_stats += template.format("Date of Birth: ".ljust(20), dob)
    driver_stats += template.format("Nationality: ".ljust(20), nationality)
    if num != -1:
        driver_stats += template.format("Number: ".ljust(20), num)
    driver_stats += template.format("Active Years: ".ljust(20), years_active)
    driver_stats += template.format("Teams: ".ljust(20), teams)
    driver_stats += template.format("Entries: ".ljust(20), num_races)
    if num_championships == 0:
        driver_stats += template.format("Highest WDC Finish: ".ljust(20), championships_str)
    else:
        driver_stats += template.format("Championships: ".ljust(20), championships_str)
    driver_stats += template.format("Wins: ".ljust(20), num_wins_str)
    driver_stats += template.format("Podiums: ".ljust(20), num_podiums_str)
    driver_stats += template.format("Career Points: ".ljust(20), career_points)
    driver_stats += template.format("Avg. Start Pos.: ".ljust(20), mean_sp)
    driver_stats += template.format("Avg. Finish Pos.: ".ljust(20), mean_fp)
    driver_stats += template.format("First Entry: ".ljust(20), first_race_name)
    if len(wins) > 0:
        driver_stats += template.format("First Win: ".ljust(20), first_win_name)
        driver_stats += template.format("Last Win: ".ljust(20), last_win_name)
    driver_stats += template.format("Last Entry: ".ljust(20), last_race_name)

    return Div(text=driver_stats)


def generate_error_layout():
    """
    Generates an error layout in the event that the user selects an invalid driver.
    :return: Error layout
    """
    text = "Somehow, you have selected an invalid driver. The drivers we have data on are..."
    text += "<ul>"
    for did in drivers.index:
        text += f"<li>{get_driver_name(did)}</li>"
    text += "</ul><br>"
    return Div(text=text)
