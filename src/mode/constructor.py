import logging
import math
import re
from collections import defaultdict, OrderedDict
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import Div, Spacer, CrosshairTool, Range1d, FixedTicker, LegendItem, Legend, HoverTool, TableColumn, \
    DataTable, ColumnDataSource, LinearAxis, NumeralTickFormatter, Span, Label, Title, DatetimeTickFormatter, Button, \
    Slider
from bokeh.plotting import figure
from pandas import Series
from data_loading.data_loader import load_constructors, load_results, load_constructor_standings, load_races, \
    load_fastest_lap_data, load_driver_standings
from mode import driver
from utils import get_constructor_name, PLOT_BACKGROUND_COLOR, position_text_to_str, get_driver_name, \
    get_circuit_name, int_to_ordinal, get_status_classification, rounds_to_str, nationality_to_flag, get_race_name, \
    millis_to_str, DATETIME_TICK_KWARGS, rescale, result_to_str
import numpy as np

constructors = load_constructors()
results = load_results()
constructor_standings = load_constructor_standings()
races = load_races()
fastest_lap_data = load_fastest_lap_data()
driver_standings = load_driver_standings()


def get_layout(constructor_id=-1, **kwargs):
    if constructor_id not in constructors.index:
        return generate_error_layout()

    constructor_results = results[results["constructorId"] == constructor_id]
    constructor_constructor_standings = constructor_standings[constructor_standings["constructorId"] == constructor_id]
    constructor_rids = constructor_results["raceId"].unique()
    constructor_races = races[races.index.isin(constructor_rids)].sort_values(by=["year", "raceId"])
    constructor_years = constructor_races[constructor_races.index.isin(constructor_constructor_standings["raceId"])]
    constructor_years = constructor_years["year"].unique()
    constructor_fastest_lap_data = fastest_lap_data[fastest_lap_data["constructor_id"] == constructor_id]
    c_driver_standings_idxs = []
    for idx, results_row in constructor_results.iterrows():
        rid = results_row["raceId"]
        did = results_row["driverId"]
        driver_standings_slice = driver_standings[(driver_standings["raceId"] == rid) &
                                                  (driver_standings["driverId"] == did)]
        c_driver_standings_idxs.extend(driver_standings_slice.index.values.tolist())
    constructor_driver_standings = driver_standings.loc[c_driver_standings_idxs]
    logging.info(f"Generating layout for mode CONSTRUCTOR in constructor, constructor_id={constructor_id}")

    if len(constructor_years) == 0:
        return Div(text="Unfortunately, this team competed for less than 1 year and thus we cannot accurately "
                        "provide data on them.")

    # Position plot
    positions_plot, positions_source = generate_positions_plot(constructor_years, constructor_constructor_standings,
                                                               constructor_results, constructor_fastest_lap_data,
                                                               constructor_id)

    # Positions bar plot
    positions_bar_plot = generate_finishing_positions_bar_plot(constructor_results)

    # WCC bar plot
    wcc_bar_plot = generate_wcc_position_bar_plot(positions_source)

    # Win plot
    win_plot = generate_win_plot(positions_source, constructor_id)

    # Teammate comparison plot
    args = [constructor_results, constructor_races, constructor_driver_standings, constructor_years]
    teammate_comparison_line_plot, comparison_source = generate_teammate_comparison_line_plot(*args)

    # Circuit performance table
    circuit_performance_table = generate_circuit_performance_table(constructor_results, constructor_races,
                                                                   constructor_id, consider_up_to=24)

    # Driver performance table
    driver_performance_layout, driver_performance_source = generate_driver_performance_table(constructor_races,
                                                                                             constructor_results)

    # Constructor stats div
    constructor_stats = generate_stats_layout(constructor_years, constructor_races,
                                              driver_performance_source, constructor_results,
                                              constructor_constructor_standings, constructor_id)

    # Header
    constructor_name = get_constructor_name(constructor_id)
    header = Div(text=f"<h2><b>{constructor_name}</b></h2><br>")

    middle_spacer = Spacer(width=5, background=PLOT_BACKGROUND_COLOR)
    layout = column([header,
                     positions_plot, middle_spacer,
                     positions_bar_plot, middle_spacer,
                     wcc_bar_plot, middle_spacer,
                     win_plot,
                     teammate_comparison_line_plot,
                     circuit_performance_table,
                     driver_performance_layout,
                     constructor_stats],
                    sizing_mode="stretch_width")

    logging.info("Finished generating layout for mode CONSTRUCTOR")

    return layout


def generate_positions_plot(constructor_years, constructor_constructor_standings, constructor_results,
                            constructor_fastest_lap_data, cid, smoothing_alpha=0.05, minor_line_width=1.7,
                            major_line_width=2.8, smoothing_muted=False, show_driver_changes=False,
                            return_components_and_source=False, include_lap_times=False, races_sublist=None,
                            show_mean_finish_pos=False):
    """
    Plots WCC standing (calculated per-race and per-season), average quali rank (per-race), average fastest lap rank
    (per-race), and average finishing position (per-race) all on the same plot. Also marks driver changes with a line.
    :param constructor_years: Constructor years
    :param constructor_constructor_standings: Constructor constructor standings
    :param constructor_results: Constructor results
    :param constructor_fastest_lap_data: Constructor fastest lap data
    :param cid: Constructor ID
    :param smoothing_alpha: Alpha used for smoothing, 0.01=very smoothed, 0.99=almost no smoothing
    :param minor_line_width: Line width for the less important lines
    :param major_line_width: Line width for the more important lines
    :param smoothing_muted: If True, the smoothed lines will be muted by default, otherwise the main lines will be muted
    :param show_driver_changes: If set to True, will show the driver changes line, if set to False will have a checkbox
    :param return_components_and_source: If set to True, will return the individual components
    :param include_lap_times: If set to True, will plot average lap time of every race
    :param races_sublist: This is an option to only use a certain sub-set of races if, for example, the user wishes to
    show positions at just a specific circuit, set to None if looking at all races
    :param show_mean_finish_pos: If set to True, will show mean finish position that year instead of WDC finish pos
    :return: Position plot layout
    """
    # TODO add smoothing slider
    logging.info("Generating position plot")
    if constructor_years.shape[0] == 0:
        return Div(text="Unfortunately, we don't have data on this constructor."), {}
    source = pd.DataFrame(columns=[
        "x",
        "constructor_name",
        "driver_names",
        "finish_position_str", "finish_position_int", "finish_positions",
        "race_id",
        "year",
        "grid",
        "avg_lap_rank",
        "points",
        "roundNum", "roundName",
        "wcc_current_standing", "wcc_current_standing_str",
        "wcc_final_standing", "wcc_final_standing_str",
        "num_dnfs_this_race",
        "avg_lap_time_millis", "avg_lap_time_str",
        "avg_finish_pos", "avg_finish_pos_str"
    ])
    name = get_constructor_name(cid)
    min_year = constructor_years.min()
    max_year = constructor_years.max()
    positions_plot = figure(
        title=u"Finishing Positions and Ranks \u2014 " + name,
        y_axis_label="Position",
        x_axis_label="Year",
        y_range=Range1d(0, 22, bounds=(0, 60)),
        tools="pan,xbox_zoom,xwheel_zoom,reset,box_zoom,wheel_zoom,save"
    )
    if show_driver_changes:
        subtitle = "Only drivers who drove with this constructor for more than 10 races are shown in driver changes"
        positions_plot.add_layout(Title(text=subtitle, text_font_style="italic"), "above")
    prev_drivers = set()
    points = 0
    round_num = 1
    driver_changes_glyphs = []
    round_name = ""
    for year in range(min_year, max_year + 1):
        if races_sublist is None:
            year_subraces = races[races["year"] == year]
        else:
            year_subraces = races_sublist[races_sublist["year"] == year]
        if year_subraces.shape[0] == 0:
            continue
        year_c_results = constructor_results[constructor_results["raceId"].isin(year_subraces.index)]
        year_fastest_lap_data = constructor_fastest_lap_data[
            constructor_fastest_lap_data["raceId"].isin(year_subraces.index)]
        dx = 1 / year_subraces.shape[0]
        x = year
        year_races = races[races["year"] == year]
        final_rid = year_races[year_races["round"] == year_races["round"].max()].index.values[0]
        final_standing = constructor_constructor_standings[constructor_constructor_standings["raceId"] == final_rid]
        year_avg_finish_pos = constructor_results[
            constructor_results["raceId"].isin(year_races.index)]["positionOrder"].mean()
        final_standing = final_standing["position"]
        if final_standing.shape[0] == 0:
            final_standing = np.nan
            final_standing_str = ""
        else:
            final_standing = final_standing.values[0]
            final_standing_str = int_to_ordinal(final_standing)
        for race_id in year_subraces.sort_index().index:
            current_standing = constructor_constructor_standings[constructor_constructor_standings["raceId"] == race_id]
            race_results = year_c_results[year_c_results["raceId"] == race_id]
            num_dnfs = race_results[race_results["positionText"].str.match("r", flags=re.IGNORECASE)].shape[0]
            if race_results.shape[0] > 0:
                driver_names = ", ".join(race_results["driverId"].apply(get_driver_name))
                constructor_name = get_constructor_name(race_results["constructorId"].values[0])
                finish_position_str = ", ".join(race_results["positionText"].apply(position_text_to_str))
                finish_positions = race_results["positionOrder"].astype(int)
                finish_position_int = round(finish_positions.mean(), 1)
                grid = round(race_results["grid"].values[0].mean(), 1)
            else:
                driver_names = ""
                constructor_name = get_constructor_name(cid)
                finish_position_str = ""
                finish_position_int = np.nan
                finish_positions = []
                grid = np.nan
            if current_standing.shape[0]:
                current_wcc_standing = current_standing["position"].values[0]
                current_wcc_standing_str = int_to_ordinal(current_wcc_standing)
                round_num = current_standing["roundNum"].values[0]
                round_name = current_standing["roundName"].values[0]
                points = current_standing["points"].values[0]
            else:
                current_wcc_standing = np.nan
                current_wcc_standing_str = ""
            race_fastest_lap_data = year_fastest_lap_data[year_fastest_lap_data["raceId"] == race_id]
            if race_fastest_lap_data.shape[0] > 0:
                avg_lap_rank = race_fastest_lap_data["avg_lap_time_rank"]
                avg_lap_rank = avg_lap_rank.astype(float).mean()
                avg_lap_rank = round(avg_lap_rank, 1)
                avg_lap_time_millis = race_fastest_lap_data["avg_lap_time_millis"].mean()
                avg_lap_time_str = millis_to_str(avg_lap_time_millis)
            else:
                avg_lap_rank = np.nan
                avg_lap_time_millis = np.nan
                avg_lap_time_str = ""
            source = source.append({
                "x": x,
                "driver_names": driver_names,
                "constructor_name": constructor_name,
                "finish_position_str": finish_position_str,
                "finish_position_int": finish_position_int,
                "finish_positions": finish_positions,
                "race_id": race_id,
                "year": year,
                "grid": grid,
                "avg_lap_rank": avg_lap_rank,
                "wcc_current_standing": current_wcc_standing,
                "wcc_current_standing_str": current_wcc_standing_str,
                "wcc_final_standing": final_standing,
                "wcc_final_standing_str": final_standing_str,
                "points": points,
                "roundNum": round_num,
                "roundName": round_name,
                "num_dnfs_this_race": num_dnfs,
                "avg_lap_time_millis": avg_lap_time_millis,
                "avg_lap_time_str": avg_lap_time_str,
                "avg_finish_pos": year_avg_finish_pos,
                "avg_finish_pos_str": "" if np.isnan(year_avg_finish_pos) else str(round(year_avg_finish_pos, 1))
            }, ignore_index=True)

            # Mark driver changes
            curr_drivers = set(race_results["driverId"])
            if prev_drivers != curr_drivers:
                new_drivers = curr_drivers - prev_drivers
                y = 10.8
                dy = -0.7
                for did in new_drivers:
                    if constructor_results[constructor_results["driverId"] == did].shape[0] > 10:
                        line = Span(line_color="white", location=x, dimension="height", line_alpha=0.7, line_width=2.3)
                        line.visible = show_driver_changes
                        label = Label(x=x + 0.03, y=y, text=get_driver_name(did, include_flag=False, just_last=True),
                                      render_mode="canvas", text_color="white", text_font_size="10pt", text_alpha=0.9,
                                      angle=math.pi / 8)
                        positions_plot.add_layout(line)
                        positions_plot.add_layout(label)
                        driver_changes_glyphs.append(line)
                        driver_changes_glyphs.append(label)
                        y += dy
            prev_drivers = curr_drivers
            x += dx

    # Apply some smoothing
    source["grid_smoothed"] = source["grid"].ewm(alpha=smoothing_alpha).mean()
    source["finish_position_int_smoothed"] = source["finish_position_int"].ewm(alpha=smoothing_alpha).mean()
    source["avg_lap_rank_smoothed"] = source["avg_lap_rank"].ewm(alpha=smoothing_alpha).mean()
    source["grid"] = source["grid"].fillna("")
    source["finish_position_int"] = source["finish_position_int"].fillna(np.nan)
    source["avg_lap_rank_str"] = source["avg_lap_rank"].fillna("")

    source = source.sort_values(by=["year", "race_id"])
    constructor_name = get_constructor_name(cid)

    min_x = source["x"].min()
    max_x = source["x"].max()
    positions_plot.x_range = Range1d(min_x, max_x, bounds=(min_x, max_x + 3))
    if len(constructor_years) > 15:
        positions_plot.xaxis.ticker = FixedTicker(ticks=np.arange(min_year - 1, max_year + 2, 3))
    else:
        positions_plot.xaxis.ticker = FixedTicker(ticks=np.arange(min_year - 1, max_year, 3))
    positions_plot.yaxis.ticker = FixedTicker(ticks=np.arange(5, 60, 5).tolist() + [1])

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
    elif races_sublist is None:
        final_standing_line = positions_plot.line(y="wcc_final_standing", color="white", line_width=major_line_width,
                                                  line_alpha=0.7, **kwargs)
        wcc_finish_position_line = positions_plot.line(y="wcc_current_standing", color="green",
                                                       line_width=major_line_width,
                                                       line_alpha=0.7, **kwargs)
        legend.extend([LegendItem(label="WCC Final Year Standing", renderers=[final_standing_line]),
                       LegendItem(label="WCC Current Standing", renderers=[wcc_finish_position_line])])
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

    legend.extend([LegendItem(label="Avg. Race Finish Position", renderers=[finish_position_line]),
                   LegendItem(label="Finish Pos. Smoothed", renderers=[finish_position_smoothed_line]),
                   LegendItem(label="Avg. Lap Time Rank", renderers=[avg_lap_rank_line]),
                   LegendItem(label="Avg. Lap Rank Smoothed", renderers=[avg_lap_rank_smoothed_line]),
                   LegendItem(label="Mean Grid Position", renderers=[grid_line]),
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

    positions_plot.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Name", constructor_name),
        ("Driver(s)", "@driver_names"),
        ("Year", "@year"),
        ("Round", "@roundNum - @roundName"),
        ("Grid Position", "@grid"),
        ("Fastest Lap Rank", "@fastest_lap_rank_str"),
        ("Finishing Position", "@finish_position_str"),
        ("Points", "@points"),
        ("Final Position this year", "@wcc_final_standing_str"),
        ("Current WCC Position", "@wcc_current_standing_str"),
        ("Avg. Lap Time", "@avg_lap_time_str"),
        ("Avg. Finish Pos. this year", "@avg_finish_pos_str")
    ]))

    # Crosshair tooltip
    positions_plot.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

    # Enable/disable the driver changes glyphs
    button = Button(label="Show Driver Changes", default_size=200)

    def update_driver_changes_visible(showing_changes=None):
        if len(driver_changes_glyphs) > 0 and showing_changes is None:
            showing_changes = not driver_changes_glyphs[0].visible
        if showing_changes is None:
            return
        for g in driver_changes_glyphs:
            g.visible = showing_changes
        if showing_changes:
            button.label = "Hide Driver Changes"
        else:
            button.label = "Show Driver Changes"
    button.on_click(lambda event: update_driver_changes_visible())
    update_driver_changes_visible(show_driver_changes)

    if show_driver_changes:
        if return_components_and_source:
            return positions_plot, source
        else:
            return column([positions_plot], sizing_mode="stretch_width"), source
    else:
        if return_components_and_source:
            return button, positions_plot, source
        else:
            return column([row([button], sizing_mode="fixed"), positions_plot], sizing_mode="stretch_width"), source


def generate_circuit_performance_table(constructor_results, constructor_races, constructor_id, consider_up_to=24):
    """
    Generates a table of the constructors's performance at every circuit, ranked by number of wins then number of 2nd
    places, then number of 3rds, etc.
    :param constructor_results: Constructor results
    :param constructor_races: Constructor races
    :param constructor_id: Constructor ID
    :param consider_up_to: Place to consider up to
    :return: Table layout
    """
    logging.info("Generating circuit performance table")
    if constructor_results.shape[0] == 0:
        return Div(text="")
    # Calculate best circuits by first ranking by number of wins, then number of 2nd places, then 3rd, etc
    circuit_scores = defaultdict(lambda: np.zeros(consider_up_to))
    circuit_years = defaultdict(lambda: [[] for _ in range(0, consider_up_to)])
    for idx, results_row in constructor_results.iterrows():
        circuit_id = constructor_races.loc[results_row["raceId"], "circuitId"]
        year = constructor_races.loc[results_row["raceId"], "year"]
        pos = results_row["positionOrder"]
        to_add = np.zeros(consider_up_to)
        if pos <= consider_up_to:
            to_add[pos - 1] = 1
            circuit_years[circuit_id][pos - 1].append(year)
        circuit_scores[circuit_id] += to_add
    circuit_scores = pd.DataFrame.from_dict(circuit_scores, orient="index")
    circuit_scores.index.name = "circuitId"
    circuit_scores = circuit_scores.sort_values(by=list(range(0, consider_up_to)), ascending=False)

    source = pd.DataFrame(columns=["circuit_name", "wins", "podiums", "other_places"])
    for cid, scores_row in circuit_scores.iterrows():
        circuit_name = get_circuit_name(cid)
        wins = scores_row[0]
        wins_years = circuit_years[cid][0]
        wins_str = str(int(wins)).rjust(2)
        if int(wins) > 0:
            wins_str += " (" + rounds_to_str(wins_years) + ")"
        podiums = scores_row[0] + scores_row[1] + scores_row[2]
        podiums_years = list(set(wins_years + circuit_years[cid][1] + circuit_years[cid][2]))
        podiums_str = str(int(podiums)).rjust(2)
        if int(podiums) > 0:
            podiums_str += " (" + rounds_to_str(podiums_years) + ")"
        other_places = ""
        place = 4
        for num_places in scores_row.values[3:10]:
            num_places = int(num_places)
            if num_places > 0:
                ordinal = int_to_ordinal(place)
                other_places += str(num_places).rjust(2) + " " + ordinal + ("s" if num_places > 1 else "") + ", "
            place += 1
        other_places = other_places[:-2] if len(other_places) > 0 else other_places
        source = source.append({
            "circuit_name": circuit_name,
            "wins": wins_str,
            "podiums": podiums_str,
            "other_places": other_places
        }, ignore_index=True)

    best_circuits_columns = [
        TableColumn(field="circuit_name", title="Circuit Name", width=150),
        TableColumn(field="wins", title="Wins", width=160),
        TableColumn(field="podiums", title="Podiums", width=200),
        TableColumn(field="other_places", title="Other Finishes (4th-10th)", width=260),
    ]

    circuits_table = DataTable(source=ColumnDataSource(data=source), columns=best_circuits_columns,
                               index_position=None, min_height=530)
    constructor_name = get_constructor_name(constructor_id, include_flag=False)
    title = Div(text=f"<h2><b>What were {constructor_name}'s Best Circuits?</b></h2>")

    return column([title, row([circuits_table], sizing_mode="stretch_width")], sizing_mode="stretch_width")


def generate_finishing_positions_bar_plot(constructor_results, consider_up_to=24):
    """
    Bar plot showing distribution of race finishing positions.
    :param constructor_results: Constructor results
    :param consider_up_to: Place to consider up to
    :return: Finishing position distribution plot layout
    """
    return driver.generate_finishing_position_bar_plot(constructor_results)


def generate_wcc_position_bar_plot(positions_source, consider_up_to=12, plot_height=400, color="white"):
    """
    Bar plot showing distribution of WCC finishing positions.
    :param positions_source: Position source
    :param consider_up_to: Position to plot up to
    :param plot_height: Plot height
    :param color: Bar colors
    :return: WCC position distribution plot layout
    """
    logging.info("Generating WCC finishing position bar plot")
    if isinstance(positions_source, dict):
        return Div(text="")
    results = positions_source.set_index("year")
    results = results.loc[~results.index.duplicated(keep="last")]
    positions = results.copy()
    n = results.shape[0]
    if n == 0:
        return Div(text="")
    results = results["wcc_final_standing"].apply(int_to_ordinal).value_counts()
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
        years_this_pos = positions[positions["wcc_final_standing"].fillna(-1).apply(int_to_ordinal) ==
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
    wcc_finish_position_plot = figure(x_range=names,
                                      title=u"WCC Finishing Positions",
                                      tools="hover",
                                      tooltips="@name: @count @years",
                                      plot_height=plot_height)
    wcc_finish_position_plot.yaxis.ticker = FixedTicker(ticks=np.arange(0, 500))
    wcc_finish_position_plot.y_range.start = 0
    wcc_finish_position_plot.y_range.end = max(counts) + 1
    wcc_finish_position_plot.vbar(x="name", top="count", width=0.9, source=source, color=color, fill_alpha=0.8,
                                  line_alpha=0.9)

    # Add the other y axis
    max_y = wcc_finish_position_plot.y_range.end
    y_range = Range1d(start=0, end=max_y / n)
    wcc_finish_position_plot.extra_y_ranges = {"percent_range": y_range}
    axis = LinearAxis(y_range_name="percent_range")
    axis.formatter = NumeralTickFormatter(format="0.0%")
    wcc_finish_position_plot.add_layout(axis, "right")

    return wcc_finish_position_plot


def generate_driver_performance_table(constructor_races, constructor_results):
    """
    Generates a table showing the following during whole existence of the constructor and per-driver:
    - Dates with that driver
    - Highest WCC finish
    - Num races
    - Num poles and % pole
    - Num wins and win %
    - Num podiums and podium %
    - Num finishes and finish %
    - Num mechanical-related DNFs and mechanical DNF %
    - Num crash-related DNFs and crash DNF %
    - Points scored and points per race
    :param constructor_races: Constructor races
    :param constructor_results: Constructor results
    :return: Driver performance layout, source
    """
    logging.info("Generating team performance table")
    if constructor_results.shape[0] == 0:
        return Div(text=""), pd.DataFrame()
    source = pd.DataFrame(columns=["driver_name",
                                   "years",
                                   "num_races",
                                   "num_poles", "num_wins", "num_podiums",
                                   "num_finishes", "num_mechanical_dnf", "num_crash_dnf",
                                   "total_points"]).set_index("driver_name")
    for driver_id in constructor_results["driverId"].unique():
        driver_name = get_driver_name(driver_id)
        driver_results = constructor_results[constructor_results["driverId"] == driver_id]
        driver_rids = driver_results["raceId"]
        years = constructor_races[constructor_races.index.isin(driver_rids)]["year"].unique().tolist()
        num_races = driver_results.shape[0]
        num_poles = driver_results[driver_results["grid"] == 1].shape[0]
        num_wins = driver_results[driver_results["positionOrder"] == 1].shape[0]
        num_podiums = driver_results[driver_results["positionOrder"] <= 3].shape[0]
        statuses = driver_results["statusId"].apply(get_status_classification)
        num_finishes = statuses[statuses.str.match("finished")].shape[0]
        num_mechanical_dnf = statuses[statuses.str.match("mechanical")].shape[0]
        num_crash_dnf = statuses[statuses.str.match("crash")].shape[0]
        total_points = driver_results["points"].sum()
        source.loc[driver_name] = {
            "years": years,
            "num_races": num_races,
            "num_poles": num_poles,
            "num_wins": num_wins,
            "num_podiums": num_podiums,
            "num_finishes": num_finishes,
            "num_mechanical_dnf": num_mechanical_dnf,
            "num_crash_dnf": num_crash_dnf,
            "total_points": total_points
        }

    # Add the all career column
    years = []
    for idx, source_row in source.iterrows():
        years.extend(source_row["years"])
    source = source.sort_values(by="years", ascending=False)
    source.loc["Total"] = {
        "years": list(set(years)),
        "num_races": constructor_races.shape[0],
        "num_poles": source["num_poles"].sum(),
        "num_wins": source["num_wins"].sum(),
        "num_podiums": source["num_podiums"].sum(),
        "num_finishes": source["num_finishes"].sum(),
        "num_mechanical_dnf": source["num_mechanical_dnf"].sum(),
        "num_crash_dnf": source["num_crash_dnf"].sum(),
        "total_points": source["total_points"].sum()
    }
    source = source.reindex(["Total"] + source.index.values.tolist()[:-1])

    source["total_points"] = source["total_points"].apply(lambda p: p if abs(int(p) - p) > 0.01 else int(p))
    source["years_str"] = source["years"].apply(lambda x: rounds_to_str(x, 1000))
    source["dnfs"] = source["num_races"] - source["num_finishes"]

    # Add percents
    def add_pct(source_row, column):
        if source_row["num_races"] == 0:
            pct = 0
        else:
            pct = 100 * source_row[column] / source_row["num_races"]
        return (str(source_row[column]) + f" ({str(round(pct, 1))}%)").rjust(11)

    def add_ppr(source_row, column):
        if source_row["num_races"] == 0:
            pct = 0
        else:
            pct = source_row[column] / source_row["num_races"]
        pts = source_row[column]
        if abs(int(pts) - pts) < 0.01:
            pts = int(pts)
        return (str(pts) + f" ({str(round(pct, 1))} pts/race)").rjust(23)

    source["num_poles_str"] = source.apply(lambda r: add_pct(r, "num_poles"), axis=1)
    source["num_wins_str"] = source.apply(lambda r: add_pct(r, "num_wins"), axis=1)
    source["num_podiums_str"] = source.apply(lambda r: add_pct(r, "num_podiums"), axis=1)
    source["num_finishes_str"] = source.apply(lambda r: add_pct(r, "num_finishes"), axis=1)
    source["num_mechanical_dnf_str"] = source.apply(lambda r: add_pct(r, "num_mechanical_dnf"), axis=1)
    source["num_crash_dnf_str"] = source.apply(lambda r: add_pct(r, "num_crash_dnf"), axis=1)
    source["total_points_str"] = source.apply(lambda r: add_ppr(r, "total_points"), axis=1)

    driver_performance_columns = [
        TableColumn(field="driver_name", title="Driver Name", width=150),
        TableColumn(field="years_str", title="Years", width=100),
        TableColumn(field="num_races", title="Races", width=50),
        TableColumn(field="num_poles_str", title="Poles", width=75),
        TableColumn(field="num_wins_str", title="Wins", width=75),
        TableColumn(field="num_podiums_str", title="Podiums", width=75),
        TableColumn(field="num_finishes_str", title="Finishes", width=75),
        TableColumn(field="num_mechanical_dnf_str", title="Mech. DNFs", width=75),
        TableColumn(field="num_crash_dnf_str", title="Crash DNFs", width=75),
        TableColumn(field="total_points_str", title="Total Points", width=100),
    ]

    driver_performance_columns_table = DataTable(source=ColumnDataSource(data=source),
                                                 columns=driver_performance_columns, index_position=None,
                                                 height=min(35 * source.shape[0], 570))
    title = Div(text=f"<h2><b>Performance of each Driver</b></h2><br><i>Again, wins, podiums, and finish percentages "
                     f"are calculated as a percentage of races entered, so they may exceeed 100% in certain cases.")

    return column([title, row([driver_performance_columns_table], sizing_mode="stretch_width")],
                  sizing_mode="stretch_width"), source


def generate_win_plot(positions_source, cid):
    """
    Plots number of races, win percentage, number of wins, podium percentage, number of podiums, number of DNFs, and
    DNF percent on the same plot.
    (2 different axes on each side).
    :param positions_source: Positions source
    :param cid: Constructor ID
    :return: Plot layout
    """
    # TODO add support for top-n finishes and calculate n
    # TODO add support for points and points per race
    return driver.generate_win_plot(positions_source, constructor_id=cid)


def generate_teammate_comparison_line_plot(constructor_results, constructor_races, constructor_driver_standings,
                                           constructor_years, return_components_and_source=False, smoothed_muted=True,
                                           highlight_driver_changes=False):
    """
    Teammate comparison line plot.
    :param constructor_results: Constructor results
    :param constructor_races: Constructor races
    :param constructor_driver_standings: Constructor driver standings
    :param constructor_years: Constructor years
    :param return_components_and_source: If True, will return slider, plot, source
    :param smoothed_muted: If True, smoothed lines will be muted by default
    :param highlight_driver_changes: If True, driver change lines will be drawn
    :return: Layout, source or slider, plot, source depending on `return_components_and_source`
    """
    # TODO add mean lap time percent to this plot
    logging.info("Generating teammate finish pos. vs driver finish pos line plot")
    source = pd.DataFrame(columns=["x", "year",
                                   "driver1_fp", "driver2_fp",
                                   "driver1_fp_str", "driver2_fp_str",
                                   "driver1_wdc_final_standing", "driver2_wdc_final_standing",
                                   "driver1_name", "driver2_name",
                                   "roundNum", "roundName"])

    teammate_fp_plot = figure(title=u"Teammate Comparison Over Time \u2014 Horizontal lines show mean finish position, "
                                    u"include DNFs",
                              y_axis_label="Finish Position Difference (Driver - Teammate)",
                              y_range=Range1d(0, 22, bounds=(0, 60)),
                              tools="pan,box_zoom,reset,save")

    prev_drivers = set()
    round_num = 1
    for year in constructor_years:
        year_races = constructor_races[constructor_races["year"] == year]
        year_results = constructor_results[constructor_results["raceId"].isin(year_races.index)]
        x = year
        if year_races.shape[0] == 0:
            continue
        dx = 1 / year_races.shape[0]
        final_rid = year_races[year_races["round"] == year_races["round"].max()].index.values[0]
        for rid in year_results["raceId"].unique():
            race_results = year_results[year_results["raceId"] == rid]
            dids = sorted(race_results["driverId"].unique().tolist())

            def get_info(did):
                driver_name = get_driver_name(did)
                driver_results = race_results[race_results["driverId"] == did]
                if driver_results.shape[0] > 0:
                    driver_fp = driver_results["positionOrder"].values[0]
                    driver_fp_str, _ = result_to_str(driver_fp, driver_results["statusId"].values[0])
                    final_standing = constructor_driver_standings[(constructor_driver_standings["raceId"] == final_rid)
                                                                  & (constructor_driver_standings["driverId"] == did)]
                    final_standing = final_standing["position"]
                    if final_standing.shape[0] == 0:
                        final_standing = -1
                    else:
                        final_standing = int_to_ordinal(final_standing.values[0])
                    return driver_name, driver_fp, driver_fp_str, final_standing
                else:
                    return "", np.nan, "", -1

            if len(dids) > 0:
                driver1_name, driver1_fp, driver1_fp_str, driver1_wdc_final_standing = get_info(dids[0])
            else:
                driver1_name = ""
                driver1_fp = np.nan
                driver1_fp_str = ""
                driver1_wdc_final_standing = ""
            if len(dids) > 1:
                driver2_name, driver2_fp, driver2_fp_str, driver2_wdc_final_standing = get_info(dids[1])
            else:
                driver2_name = ""
                driver2_fp = np.nan
                driver2_fp_str = ""
                driver2_wdc_final_standing = ""
            source = source.append({
                "x": x,
                "driver1_fp": driver1_fp,
                "driver2_fp": driver2_fp,
                "driver1_fp_str": driver1_fp_str,
                "driver2_fp_str": driver2_fp_str,
                "driver1_wdc_final_standing": driver1_wdc_final_standing,
                "driver2_wdc_final_standing": driver2_wdc_final_standing,
                "driver1_name": driver1_name,
                "driver2_name": driver2_name,
                "roundNum": str(round_num),
                "roundName": get_race_name(rid)
            }, ignore_index=True)
            x += dx
            round_num += 1

            # Mark driver changes
            curr_drivers = set(dids)
            if prev_drivers != curr_drivers:
                new_drivers = curr_drivers - prev_drivers
                y = 18
                dy = -1.5
                x_offset = 0.1 if len(constructor_years) > 5 else 0.02
                for did in new_drivers:
                    alpha = 0.7 if highlight_driver_changes else 0.1
                    line = Span(line_color="white", location=x, dimension="height", line_alpha=alpha, line_width=3.2)
                    if highlight_driver_changes:
                        label = Label(x=x + x_offset, y=y, text=get_driver_name(did, include_flag=False, just_last=True),
                                      render_mode="canvas", text_color="white", text_font_size="12pt", text_alpha=0.9,
                                      angle=math.pi / 8)
                        teammate_fp_plot.add_layout(label)
                    teammate_fp_plot.add_layout(line)
                    y += dy
            prev_drivers = curr_drivers
    source["fp_diff"] = source["driver1_fp"] - source["driver2_fp"]

    source["driver1_fp_smoothed"] = source["driver1_fp"].ewm(alpha=0.05).mean()
    source["driver2_fp_smoothed"] = source["driver2_fp"].ewm(alpha=0.05).mean()

    column_source = ColumnDataSource(data=source)

    subtitle = "Note that with driver changes (indicated by white vertical line), driver 1 and driver 2 may swap."
    teammate_fp_plot.add_layout(Title(text=subtitle, text_font_style="italic"), "above")

    min_x = source["x"].min()
    max_x = source["x"].max()
    teammate_fp_plot.x_range = Range1d(min_x, max_x, bounds=(min_x, max_x + 3))
    teammate_fp_plot.xaxis.ticker = FixedTicker(ticks=np.arange(1950, 2050))
    teammate_fp_plot.yaxis.ticker = FixedTicker(ticks=np.arange(5, 61, 5).tolist() + [1])

    kwargs = dict(
        x="x",
        source=column_source,
        line_width=2,
        muted_alpha=0
    )
    driver1_fp_line = teammate_fp_plot.line(y="driver1_fp", color="white", **kwargs)
    driver1_fp_smoothed_line = teammate_fp_plot.line(y="driver1_fp_smoothed", color="white", line_dash="dashed",
                                                     **kwargs)
    driver2_fp_line = teammate_fp_plot.line(y="driver2_fp", color="yellow", **kwargs)
    driver2_fp_smoothed_line = teammate_fp_plot.line(y="driver2_fp_smoothed", color="yellow", line_dash="dashed",
                                                     **kwargs)

    # Draw line at means
    mean_driver_fp = source["driver1_fp"].mean()
    mean_teammate_fp = source["driver2_fp"].mean()
    line_kwargs = dict(
        x=[-1000, 5000],
        line_alpha=0.4,
        line_width=2.5,
        muted_alpha=0
    )
    driver1_mean_line = teammate_fp_plot.line(line_color="white", y=[mean_driver_fp] * 2, **line_kwargs)
    driver2_mean_line = teammate_fp_plot.line(line_color="yellow", y=[mean_teammate_fp] * 2, **line_kwargs)

    if smoothed_muted:
        driver1_fp_line.muted = True
        driver2_fp_line.muted = True
        driver1_mean_line.muted = True
        driver2_mean_line.muted = True
    else:
        driver1_fp_smoothed_line.muted = True
        driver2_fp_smoothed_line.muted = True

    # Legend
    legend = [LegendItem(label="Driver 1 Finish Pos.", renderers=[driver1_fp_line, driver1_mean_line]),
              LegendItem(label="Finish Pos. Smoothed", renderers=[driver1_fp_smoothed_line]),
              LegendItem(label="Driver 2 Finish Pos.", renderers=[driver2_fp_line, driver2_mean_line]),
              LegendItem(label="Finish Pos. Smoothed", renderers=[driver2_fp_smoothed_line])]
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
            source["driver1_fp_smoothed"] = source["driver1_fp"]
            source["driver2_fp_smoothed"] = source["driver2_fp"]
        else:
            source["driver1_fp_smoothed"] = source["driver1_fp"].ewm(alpha=alpha).mean()
            source["driver2_fp_smoothed"] = source["driver2_fp"].ewm(alpha=alpha).mean()
        column_source.patch({
            "driver1_fp_smoothed": [(slice(source["driver1_fp_smoothed"].shape[0]), source["driver1_fp_smoothed"])],
            "driver2_fp_smoothed": [(slice(source["driver2_fp_smoothed"].shape[0]), source["driver2_fp_smoothed"])],
        })

    smoothing_slider = Slider(start=0, end=1, value=0.95, step=.01, title="Smoothing Amount, 0=no "
                                                                          "smoothing, 1=heavy smoothing")
    smoothing_slider.on_change("value", lambda attr, old, new: smoothing_cb(new))

    # Hover tooltip
    teammate_fp_plot.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Finish Position", "@driver1_fp"),
        ("Driver 1", "@driver1_name"),
        ("Driver 1 Finish Pos.", "@driver1_fp_str"),
        ("Driver 1 Final Pos. this year", "@driver1_wdc_final_standing"),
        ("Driver 2", "@driver2_name"),
        ("Driver 2 Finish Pos.", "@driver2_fp_str"),
        ("Driver 2 Final Pos. this year", "@driver2_wdc_final_standing"),
        ("Round", "@roundNum - @roundName"),
    ]))

    # Crosshair tooltip
    teammate_fp_plot.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

    if return_components_and_source:
        return smoothing_slider, teammate_fp_plot, source
    else:
        return column([smoothing_slider, teammate_fp_plot], sizing_mode="stretch_width"), source


def generate_stats_layout(constructor_years, constructor_races, performance_source, constructor_results,
                          constructor_constructor_standings, cid):
    """
    Includes some information not found in the "Total" category of the driver performance table.
    :param constructor_years: Constructor years
    :param constructor_races: Constructor races
    :param performance_source: Performance source
    :param constructor_results: Constructor results
    :param constructor_constructor_standings: Constructor constructor results
    :param cid: Driver ID
    :return: Stats layout
    """
    logging.info("Generating constructor stats layout")
    if performance_source.shape[0] <= 1:
        return Div(text="")
    constructor_name = get_constructor_name(cid, include_flag=False)
    constructor_entry = constructors.loc[cid]
    nationality = constructor_entry["nationality"].lower().title()
    nationality = nationality_to_flag(nationality.lower()) + nationality

    years_active = rounds_to_str(constructor_years)

    mean_sp = round(constructor_results["grid"].mean(), 1)
    mean_fp = round(constructor_results["positionOrder"].mean(), 1)

    first_year = np.min(constructor_years)
    first_rid = constructor_races[constructor_races["year"] == first_year].sort_values(by="round").index.values[0]
    first_race_name = str(first_year) + " " + get_race_name(first_rid)
    last_year = np.max(constructor_years)
    last_rid = constructor_races[constructor_races["year"] == last_year].sort_values(by="round", ascending=True)
    last_rid = last_rid.index.values[0]
    last_race_name = str(last_year) + " " + get_race_name(last_rid)

    wins = []
    constructor_results = constructor_results.set_index("raceId")
    for rid, race_row in constructor_races.iterrows():
        position = constructor_results.loc[rid, "positionOrder"]
        if isinstance(position, Series):
            position = position.values[0]
        if rid in constructor_results.index and position == 1:
            wins.append(rid)
    if len(wins) > 0:
        first_win_rid = wins[0]
        first_win_year = races.loc[first_win_rid, "year"]
        first_win_name = str(first_win_year) + " " + get_race_name(first_win_rid)
        last_win_rid = wins[-1]
        last_win_year = races.loc[first_win_rid, "year"]
        last_win_name = str(last_win_year) + " " + get_race_name(last_win_rid)

    performance_source = performance_source.loc["Total"]
    num_podiums_str = performance_source["num_podiums_str"]
    num_wins_str = performance_source["num_wins_str"]
    career_points = performance_source["total_points"]
    num_races = performance_source["num_races"]

    num_championships = 0
    wcc_final_standings_total = []
    championships_str = " ("
    for year in constructor_years:
        year_races = constructor_races[constructor_races["year"] == year]
        final_rid = year_races[year_races["round"] == year_races["round"].max()].index.values[0]
        final_standing = constructor_constructor_standings[constructor_constructor_standings["raceId"] == final_rid]
        final_standing = final_standing["position"]
        if final_standing.shape[0] > 0:
            final_standing = final_standing.values[0]
            wcc_final_standings_total.append(final_standing)
            if final_standing == 1:
                num_championships += 1
                championships_str += str(year) + ", "

    if num_championships == 0:
        championships_str = str(np.max(wcc_final_standings_total))
    else:
        championships_str = str(num_championships) + championships_str[:-2] + ")"

    classifications = constructor_results["statusId"].apply(get_status_classification)
    num_mechanical_dnfs = classifications[classifications == "mechanical"].shape[0]
    num_crash_dnfs = classifications[classifications == "crash"].shape[0]
    num_finishes = classifications[classifications == "finished"].shape[0]
    mechanical_dnfs_str = str(num_mechanical_dnfs)
    crash_dnfs_str = str(num_crash_dnfs)
    finishes_str = str(num_finishes)
    if num_races > 0:
        mechanical_dnfs_str += " (" + str(round(100 * num_mechanical_dnfs / num_races, 1)) + "%)"
        crash_dnfs_str += " (" + str(round(100 * num_crash_dnfs / num_races, 1)) + "%)"
        finishes_str += " (" + str(round(100 * num_finishes / num_races, 1)) + "%)"

    # Constructor the HTML
    header_template = """
    <h2 style="text-align: center;"><b>{}</b></h2>
    """

    template = """
    <pre><b>{}</b> {}<br></pre>
    """

    constructor_stats = header_template.format("Constructor Stats")
    constructor_stats += template.format("Name: ".ljust(20), constructor_name)
    constructor_stats += template.format("Nationality: ".ljust(20), nationality)
    constructor_stats += template.format("Active Years: ".ljust(20), years_active)
    constructor_stats += template.format("Entries: ".ljust(20), num_races)
    if num_championships == 0:
        constructor_stats += template.format("Highest WCC Finish: ".ljust(20), championships_str)
    else:
        constructor_stats += template.format("Championships: ".ljust(20), championships_str)
    constructor_stats += template.format("Wins: ".ljust(20), num_wins_str)
    constructor_stats += template.format("Podiums: ".ljust(20), num_podiums_str)
    constructor_stats += template.format("Career Points: ".ljust(20), career_points)
    constructor_stats += template.format("Avg. Start Pos.: ".ljust(20), mean_sp)
    constructor_stats += template.format("Avg. Finish Pos.: ".ljust(20), mean_fp)
    constructor_stats += template.format("First Entry: ".ljust(20), first_race_name)
    if len(wins) > 0:
        constructor_stats += template.format("First Win: ".ljust(20), first_win_name)
        constructor_stats += template.format("Last Win: ".ljust(20), last_win_name)
    constructor_stats += template.format("Last Entry: ".ljust(20), last_race_name)
    constructor_stats += template.format("Num. Mechanical DNFs: ".ljust(22), mechanical_dnfs_str)
    constructor_stats += template.format("Num. Crash DNFs: ".ljust(22), crash_dnfs_str)
    constructor_stats += template.format("Num Finishes".ljust(22), finishes_str)

    return Div(text=constructor_stats)


def generate_error_layout():
    """
    Generates an error layout in the event that the user selects an invalid constructor.
    :return: Error layout
    """
    text = "Somehow, you have selected an invalid constructor. The constructors we have data on are..."
    text += "<ul>"
    for cid in constructors.index:
        text += f"<li>{get_constructor_name(cid)}</li>"
    text += "</ul><br>"
    return Div(text=text)

