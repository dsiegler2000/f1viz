import logging
import math
import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import Div, Span, Label, Spacer, FixedTicker, Range1d, TableColumn, DataTable, ColumnDataSource, \
    CrosshairTool, HoverTool, Title, LinearAxis, NumeralTickFormatter, LegendItem, Legend
from bokeh.plotting import figure
from pandas import Series
from mode import year, driver, driverconstructor
from utils import get_driver_name, ColorDashGenerator, get_constructor_name, PLOT_BACKGROUND_COLOR, get_race_name, \
    int_to_ordinal, get_status_classification, millis_to_str, result_to_str
from data_loading.data_loader import load_driver_standings, load_results, load_races, load_fastest_lap_data, load_status

# Note, YD = year driver

driver_standings = load_driver_standings()
results = load_results()
races = load_races()
fastest_lap_data = load_fastest_lap_data()
status = load_status()

# TODO do second pass on me next :)
#  come up with 1 more plot this is looking a bit barren, maybe like comparing all races to best/worst race somehow,
#   maybe also look at variability or something?, maybe just a simple plot comparing to other drivers?

# TODO
#  Go through each existing mode and do a "second pass" to add simple features and make small clean-up changes
#   Make sure tables are sortable
#   Make sure second axes are scaled properly
#   Make sure using ordinals (1st, 2nd, 3rd) on everything
#   Make sure the mode has a header


def get_layout(year_id=-1, driver_id=-1, **kwargs):
    year_races = races[races["year"] == year_id]
    year_rids = sorted(year_races.index.values)
    year_results = results[results["raceId"].isin(year_rids)].sort_values(by="raceId")

    # Detect invalid combos
    if driver_id not in year_results["driverId"].unique():
        return generate_error_layout(year_id, driver_id)

    # Generate more slices
    year_driver_standings = driver_standings[driver_standings["raceId"].isin(year_rids)].sort_values(by="raceId")
    yd_driver_standings = year_driver_standings[year_driver_standings["driverId"] == driver_id]
    yd_results = year_results[year_results["driverId"] == driver_id]
    year_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"].isin(year_rids)]
    yd_fastest_lap_data = year_fastest_lap_data[year_fastest_lap_data["driver_id"] == driver_id]
    yd_races = year_races
    constructor_results_idxs = []
    for idx, results_row in yd_results.iterrows():
        cid = results_row["constructorId"]
        rid = results_row["raceId"]
        constructor_results_idxs.extend(year_results[(year_results["raceId"] == rid) &
                                                     (year_results["constructorId"] == cid)].index.values.tolist())
    constructor_results = year_results.loc[constructor_results_idxs]

    logging.info(f"Generating layout for mode YEARDRIVER in yeardriver, year_id={year_id}, driver_id={driver_id}")

    # More focused WDC plot
    wdc_plot = generate_wdc_plot(year_races, year_driver_standings, year_results, driver_id)

    # Positions plot
    positions_plot, positions_source = generate_positions_plot(yd_driver_standings, yd_results, yd_fastest_lap_data,
                                                               year_id, driver_id)

    # Win plot
    win_plot = generate_win_plot(positions_source, year_results)

    # Finishing position bar plot
    finishing_position_bar_plot = generate_finishing_position_bar_plot(yd_results)

    # Start pos vs finish pos scatter
    spvfp_scatter = generate_spvfp_scatter(yd_results, yd_races, yd_driver_standings)

    # Mean lap time rank vs finish pos scatter
    mltr_fp_scatter = generate_mltr_fp_scatter(yd_results, yd_races, yd_driver_standings, driver_id)

    # Teammate comparison line plot
    teammate_comparison_line_plot, comparison_source = generate_teammate_comparison_line_plot(positions_source,
                                                                                              constructor_results,
                                                                                              yd_results, driver_id)

    # Results table
    results_table = generate_results_table(yd_results, yd_fastest_lap_data, year_results, year_fastest_lap_data,
                                           driver_id)

    # Stats
    stats_layout = generate_stats_layout(positions_source, comparison_source, constructor_results, year_id, driver_id)

    middle_spacer = Spacer(width=5, background=PLOT_BACKGROUND_COLOR)
    layout = column([wdc_plot, middle_spacer,
                     positions_plot, middle_spacer,
                     win_plot, middle_spacer,
                     finishing_position_bar_plot, middle_spacer,
                     row([spvfp_scatter, mltr_fp_scatter], sizing_mode="stretch_width"), middle_spacer,
                     teammate_comparison_line_plot,
                     results_table,
                     stats_layout],
                    sizing_mode="stretch_width")

    logging.info("Finished generating layout for mode YEARDRIVER")

    return layout


def generate_wdc_plot(year_races, year_driver_standings, year_results, driver_id, consider_window=2):
    """
    Plot the championship progression plot, but only include maybe like the 5 drivers who finished near this driver
    (i.e. if he finished 7th, only include the drivers who finished 5th, 6th, 7th, 8th, and 9th)
    :param year_races: Year races
    :param year_driver_standings: Year driver standings
    :param year_results: Year results
    :param driver_id: Driver ID
    :param consider_window: Window to consider (i.e. if 2, then consider the 2 drivers above and 2 below the driver)
    :return: WDC plot layout
    """
    # Get the driver's final position
    final_rid = year_races[year_races["round"] == year_races["round"].max()].index.values[0]
    final_standings = year_driver_standings[year_driver_standings["raceId"] == final_rid].set_index("driverId")
    if driver_id in final_standings.index:
        driver_final_standing = final_standings.loc[driver_id, "position"]
        if isinstance(driver_final_standing, Series):
            driver_final_standing = driver_final_standing.values[0]
        if driver_final_standing > consider_window:
            min_position = driver_final_standing - consider_window
            max_position = driver_final_standing + consider_window
        else:
            min_position = 1
            max_position = 2 * consider_window + 1
        considering_dids = final_standings[(final_standings["position"] >= min_position) &
                                           (final_standings["position"] <= max_position)].index.unique().tolist()
        all_dids = set(final_standings.index)
        muted_dids = all_dids - set(considering_dids)
        return year.generate_wdc_plot(year_driver_standings, year_results, highlight_did=driver_id,
                                      muted_dids=muted_dids)
    return Div(text=f"Unfortunately, we have encountered an error or {get_driver_name(driver_id, include_flag=False)} "
                    f"was never officially classified in this season.")


def generate_positions_plot(yd_driver_standings, yd_results, yd_fastest_lap_data, year_id, driver_id):
    """
    Plot WDC position (both rounds and full season), quali, fastest lap, and finishing position rank vs time all on the
    same graph along with smoothed versions of these. Also marks teammate and team changes.
    :param yd_driver_standings: YD driver standings
    :param yd_results: YD results
    :param yd_fastest_lap_data: YD fastest lap data
    :param year_id: Year ID
    :param driver_id: Driver ID
    :return: Position plot, position source
    """
    driver_years = np.array([year_id])
    kwargs = dict(
        smoothing_alpha=0.2,
        smoothing_muted=True
    )
    positions_plot, positions_source = driver.generate_positions_plot(driver_years, yd_driver_standings, yd_results,
                                                                      yd_fastest_lap_data, driver_id, **kwargs)

    mark_teammate_team_changes(yd_results, positions_source, driver_id, positions_plot)

    # Add the axis overrides
    x_min = positions_source["x"].min() - 0.001
    x_max = positions_source["x"].max() + 0.001
    positions_plot.x_range = Range1d(x_min, x_max, bounds=(x_min, x_max))
    positions_plot.xaxis.ticker = FixedTicker(ticks=positions_source["x"])
    positions_plot.xaxis.major_label_overrides = {row["x"]: row["roundName"] for idx, row in
                                                  positions_source.iterrows()}
    positions_plot.xaxis.major_label_orientation = 0.8 * math.pi / 2

    return positions_plot, positions_source


def mark_teammate_team_changes(yd_results, positions_source, driver_id, fig, x_offset=0.03):
    """
    Marks team and teammate changes
    :param yd_results: YD results
    :param positions_source: Positions source
    :param driver_id: Driver ID
    :param fig: Figure to mark
    :param x_offset: Offset for drawing text labels
    :return: None
    """
    if yd_results.shape[0] > 0:
        prev_constructor = -1
        prev_teammate_did = -1
        color_gen = ColorDashGenerator()
        label_kwargs = dict(render_mode="canvas",
                            text_color="white",
                            text_font_size="10pt",
                            angle=math.pi / 4)

        def draw_mark(location, line_color, text, up=True):
            line = Span(line_color=line_color, location=location, dimension="height", line_alpha=0.4, line_width=3.2)
            fig.add_layout(line)
            label = Label(x=location + x_offset, y=18 if up else 16, text=text, **label_kwargs)
            fig.add_layout(label)

        for race_id in yd_results["raceId"]:
            cid = yd_results[yd_results["raceId"] == race_id]["constructorId"]
            x = positions_source[positions_source["race_id"] == race_id]["x"]
            if x.shape[0] > 0:
                x = x.values[0]
            else:
                continue
            if cid.shape[0] > 0:
                cid = cid.values[0]
                constructor_race_results = results[(results["raceId"] == race_id) & (results["constructorId"] == cid)]
                dids = constructor_race_results["driverId"].unique().tolist()
                if driver_id in dids:
                    dids.remove(driver_id)
                if len(dids) > 0:
                    teammate_did = dids[0]
                    if teammate_did != prev_teammate_did:
                        draw_mark(x, "white", get_driver_name(teammate_did, include_flag=False, just_last=True))
                        prev_teammate_did = teammate_did
            # Team changes
            race_results = yd_results[yd_results["raceId"] == race_id]
            if race_results.shape[0] > 0:
                curr_constructor = race_results["constructorId"].values[0]
                if curr_constructor != prev_constructor:  # Mark the constructor change
                    color, _ = color_gen.get_color_dash(None, curr_constructor)
                    draw_mark(x, color, get_constructor_name(curr_constructor, include_flag=False), up=False)
                    prev_constructor = curr_constructor


def generate_results_table(yd_results, yd_fastest_lap_data, year_results, year_fastest_lap_data, driver_id):
    """
    Table of results at each race, including quali position, finish position (or reason for DNF), time, gap to leader,
    fastest lap time and gap to fastest lap (of all drivers), average lap time and gap to fastest average lap time
    (of all drivers)
    :param yd_results: YD results
    :param yd_fastest_lap_data: YD fastest lap data
    :param year_results: Year results
    :param year_fastest_lap_data: Year fastest lap data
    :param driver_id: Driver ID
    :return: Results table
    """
    logging.info("Generating results table")
    source = pd.DataFrame(columns=["race_name", "constructor_name",
                                   "quali_pos_str",
                                   "finish_pos_str",
                                   "time_str",
                                   "fastest_lap_time_str",
                                   "avg_lap_time_str"])
    for idx, results_row in yd_results.iterrows():
        rid = results_row["raceId"]
        race_results = year_results[year_results["raceId"] == rid]
        race_fastest_lap_data = year_fastest_lap_data[year_fastest_lap_data["raceId"] == rid]
        race_driver_fastest_lap_data = yd_fastest_lap_data[yd_fastest_lap_data["raceId"] == rid]
        race_name = get_race_name(rid)
        constructor_name = get_constructor_name(results_row["constructorId"])
        quali_pos_str = int_to_ordinal(results_row["grid"])
        finish_pos = str(results_row["positionOrder"])
        status_id = results_row["statusId"]
        finish_pos_str, finish_pos = result_to_str(finish_pos, status_id)
        classification = get_status_classification(status_id)
        time = results_row["milliseconds"]
        winner = race_results[race_results["positionOrder"] == 1]
        if winner.shape[0] > 0 and winner["driverId"].values[0] != driver_id \
                and not np.isnan(time) and not np.isnan(results_row["position"]):
            time_gap = millis_to_str(time - winner["milliseconds"].values[0])
            time_str = millis_to_str(time) + " (+" + time_gap + ")"
            if status_id != 1 and classification == "finished":
                time_str = millis_to_str(time) + " (+" + time_gap + ", " + status.loc[status_id, "status"] + ")"
        elif finish_pos == 1:
            time_str = millis_to_str(time)
        else:
            time_str = ""
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
            fastest_avg_idx = race_fastest_lap_data["avg_lap_time_millis"].idxmin()
            avg_lap_time = race_driver_fastest_lap_data["avg_lap_time_millis"].values[0]
            if np.isnan(avg_lap_time):
                avg_lap_time_str = ""
            elif race_fastest_lap_data.loc[fastest_avg_idx, "driver_id"] == driver_id or np.isnan(avg_lap_time):
                avg_lap_time_str = millis_to_str(avg_lap_time) + " (Fastest Avg.)"
            else:
                fastest_avg_time = race_fastest_lap_data.loc[fastest_avg_idx, "avg_lap_time_millis"]
                avg_gap = millis_to_str(avg_lap_time - fastest_avg_time)
                avg_lap_time_str = millis_to_str(avg_lap_time) + " (+" + avg_gap + ")"
        else:
            fastest_lap_time_str = ""
            avg_lap_time_str = ""
        source = source.append({
            "race_name": race_name,
            "constructor_name": constructor_name,
            "quali_pos_str": quali_pos_str,
            "finish_pos_str": finish_pos_str,
            "time_str": time_str,
            "fastest_lap_time_str": fastest_lap_time_str,
            "avg_lap_time_str": avg_lap_time_str
        }, ignore_index=True)

    results_columns = [
        TableColumn(field="race_name", title="Race Name", width=100),
        TableColumn(field="constructor_name", title="Constructor", width=100),
        TableColumn(field="quali_pos_str", title="Grid Pos.", width=75),
        TableColumn(field="finish_pos_str", title="Finish Pos.", width=75),
        TableColumn(field="time_str", title="Time", width=100),
        TableColumn(field="fastest_lap_time_str", title="Fastest Lap Time", width=75),
        TableColumn(field="avg_lap_time_str", title="Avg. Lap Time", width=75),
    ]
    results_table = DataTable(source=ColumnDataSource(data=source), columns=results_columns, index_position=None,
                              height=27 * yd_results.shape[0])
    title = Div(text=f"<h2><b>Results for each race</b></h2><br><i>The fastest lap time and average lap time gaps "
                     f"shown are calculated based on the gap to the fastest of all drivers and fastest average of "
                     f"all drivers in that race respectively.</i>")
    return column([title, row([results_table], sizing_mode="stretch_width")], sizing_mode="stretch_width")


def generate_win_plot(positions_source, year_results):
    """
    Plot number of wins, number of podiums, number of points, win percent, podium percent, and points per race as a
    percentage of #1's points on one plot.
    Note, this is different from the regular win plot (no num races, instead points and pts per race).
    :param positions_source: Positions source
    :param year_results: Year results
    :return: Win plot layout
    """
    # TODO maybe even refactor this to use a version in driver
    # Inspired by driver.generate_win_plot
    logging.info("Generating win plot")
    if isinstance(positions_source, dict):
        return Div(text="")
    win_source = pd.DataFrame(columns=["x",
                                       "win_pct", "wins", "win_pct_str",
                                       "podium_pct", "podiums", "podium_pct_str",
                                       "ppr_pct", "points", "ppr_pct_str"
                                       "constructor_name", "wdc_final_standing"])
    wins = 0
    podiums = 0
    n_races = 0
    max_potential_points = 0
    max_potential_ppr = year_results[year_results["position"] == 1]["points"].mode().values[0]
    for idx, row in positions_source.sort_values(by="x").iterrows():
        x = row["x"]
        pos = row["finish_position_int"]
        if not np.isnan(pos):
            wins += 1 if pos == 1 else 0
            podiums += 1 if 3 >= pos > 0 else 0
        points = row["points"]
        max_potential_points += max_potential_ppr
        n_races += 1
        win_pct = wins / n_races
        podium_pct = podiums / n_races
        ppr_pct = points / max_potential_points
        win_source = win_source.append({
            "x": x,
            "wins": wins,
            "win_pct": win_pct,
            "podiums": podiums,
            "podium_pct": podium_pct,
            "points": points,
            "ppr_pct": ppr_pct,
            "constructor_name": row["constructor_name"],
            "wdc_final_standing": row["wdc_final_standing"],
            "win_pct_str": str(round(100 * win_pct, 1)) + "%",
            "podium_pct_str": str(round(100 * podium_pct, 1)) + "%",
            "ppr_pct_str": str(round(100 * ppr_pct, 1)) + "%"
        }, ignore_index=True)

    max_podium = win_source["podiums"].max()
    y_max_range = max_podium if max_podium > 0 else 10
    win_plot = figure(
        title="Wins, Podiums, and Points",
        y_axis_label="",
        x_axis_label="Year",
        tools="pan,xbox_zoom,reset,box_zoom,wheel_zoom,save",
        y_range=Range1d(0, y_max_range, bounds=(-20, 30))
    )
    subtitle = "The points percent is calculated as a percentage of the maximum number of points that driver could " \
               "achieve (if he/she won every race)"
    win_plot.add_layout(Title(text=subtitle, text_font_style="italic"), "above")

    max_podium_pct = win_source["podium_pct"].max()
    max_ppr_pct = win_source["ppr_pct"].max()
    max_points = win_source["points"].max()
    if max_ppr_pct > max_podium_pct and podiums == 0:
        k = 10 / max_ppr_pct
        max_y_pct = max_ppr_pct
    elif max_ppr_pct > max_podium_pct and max_ppr_pct > 0:
        k = max_podium / max_ppr_pct
        max_y_pct = max_ppr_pct
    elif max_podium_pct > 0:
        k = max_podium / max_podium_pct
        max_y_pct = max_podium_pct
    else:
        max_y_pct = 1
        k = 1
    win_source["podium_pct_scaled"] = k * win_source["podium_pct"]
    win_source["win_pct_scaled"] = k * win_source["win_pct"]
    win_source["ppr_pct_scaled"] = k * win_source["ppr_pct"]
    if max_points > 0 and max_podium == 0:
        k_pts = 10 / max_points
    elif max_points > 0:
        k_pts = max_podium / max_points
    else:
        k_pts = 1
        max_y_pct = 1
    win_source["points_scaled"] = k_pts * win_source["points"]

    if max_points == 0 and max_podium == 0:  # Theoretically this case could be handled, but not really useful
        return Div()

    # Override the x axis
    x_min = positions_source["x"].min() - 0.001
    x_max = positions_source["x"].max() + 0.001
    win_plot.x_range = Range1d(x_min, x_max, bounds=(x_min, x_max))
    win_plot.xaxis.ticker = FixedTicker(ticks=positions_source["x"])
    win_plot.xaxis.major_label_overrides = {row["x"]: row["roundName"] for idx, row in positions_source.iterrows()}
    win_plot.xaxis.major_label_orientation = 0.8 * math.pi / 2

    # Other y axis (%)
    max_y = win_plot.y_range.end
    y_range = Range1d(start=0, end=max_y_pct, bounds=(-0.02, 1000))
    win_plot.extra_y_ranges = {"percent_range": y_range}
    axis = LinearAxis(y_range_name="percent_range")
    axis.formatter = NumeralTickFormatter(format="0.0%")
    win_plot.add_layout(axis, "right")

    # Third y axis (points)
    y_range2 = Range1d(start=0, end=max_y / k, bounds=(-0.02, 1000))
    win_plot.extra_y_ranges["points_range"] = y_range2
    axis2 = LinearAxis(y_range_name="points_range", axis_label="Points")
    win_plot.add_layout(axis2, "right")

    kwargs = {
        "x": "x",
        "line_width": 2,
        "line_alpha": 0.7,
        "source": win_source,
        "muted_alpha": 0.01
    }
    wins_line = win_plot.line(y="wins", color="green", **kwargs)
    win_pct_line = win_plot.line(y="win_pct_scaled", color="green", line_dash="dashed", **kwargs)
    podiums_line = win_plot.line(y="podiums", color="yellow", **kwargs)
    podium_pct_line = win_plot.line(y="podium_pct_scaled", color="yellow", line_dash="dashed", **kwargs)
    points_line = win_plot.line(y="points_scaled", color="white", **kwargs)
    ppr_pct_line = win_plot.line(y="ppr_pct_scaled", color="white", line_dash="dashed", **kwargs)

    legend = [LegendItem(label="Number of Wins", renderers=[wins_line]),
              LegendItem(label="Win Percentage", renderers=[win_pct_line]),
              LegendItem(label="Number of Podiums", renderers=[podiums_line]),
              LegendItem(label="Podium Percentage", renderers=[podium_pct_line]),
              LegendItem(label="Points", renderers=[points_line]),
              LegendItem(label="Points Percentage", renderers=[ppr_pct_line])]

    legend = Legend(items=legend, location="top_right", glyph_height=15, spacing=2, inactive_fill_color="gray")
    win_plot.add_layout(legend, "right")
    win_plot.legend.click_policy = "mute"
    win_plot.legend.label_text_font_size = "12pt"

    # Hover tooltip
    win_plot.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Number of Wins", "@wins (@win_pct_str)"),
        ("Number of Podiums", "@podiums (@podium_pct_str)"),
        ("Points", "@points (@ppr_pct_str of max pts. possible)"),
        ("Constructor", "@constructor_name"),
        ("Final Position this year", "@wdc_final_standing")
    ]))

    # Crosshair tooltip
    win_plot.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

    return win_plot


def generate_finishing_position_bar_plot(yd_results):
    """
    Plot race finishing position bar chart
    :param yd_results: Year driver results
    :return: Finishing position distribution plot layout
    """
    return driver.generate_finishing_position_bar_plot(yd_results)


def generate_spvfp_scatter(yd_results, yd_races, yd_driver_standings):
    """
    Plot start position vs finish position scatter plot (see `circuitdriver.generate_spvfp_scatter`)
    :param yd_results: YD results
    :param yd_races: YD races
    :param yd_driver_standings: YD driver standings
    :return: Start pos. vs finish pos. scatter
    """
    return driver.generate_spvfp_scatter(yd_results, yd_races, yd_driver_standings, include_race_labels=True)


def generate_mltr_fp_scatter(yd_results, yd_races, yd_driver_standings, driver_id):
    """
    Plot mean lap time vs finish position scatter plot (see `driver.generate_mltr_fp_scatter`)
    :param yd_results: YD results
    :param yd_races: YD races
    :param yd_driver_standings: YD driver standings
    :param driver_id: Driver ID
    :return: Mean lap time rank vs finish pos scatter plot layout
    """
    return driver.generate_mltr_fp_scatter(yd_results, yd_races, yd_driver_standings, driver_id,
                                           include_race_labels=True)


def generate_teammate_comparison_line_plot(positions_source, constructor_results, yd_results, driver_id):
    """
    Plot finish position along with teammate finish position vs time, see
    driverconstructor.generate_teammate_comparison_line_plot
    :param positions_source: Positions source
    :param constructor_results: Constructor results
    :param yd_results: YD results
    :param driver_id: Driver ID
    :return: Teammate comparison line plot layout, source
    """
    kwargs = dict(
        return_components_and_source=True,
        default_alpha=0.5,
        mute_smoothed=True
    )
    slider, teammate_fp_plot, source = driverconstructor.generate_teammate_comparison_line_plot(positions_source,
                                                                                                constructor_results,
                                                                                                driver_id,
                                                                                                **kwargs)

    mark_teammate_team_changes(yd_results, positions_source, driver_id, teammate_fp_plot)

    # x axis override
    x_min = positions_source["x"].min() - 0.001
    x_max = positions_source["x"].max() + 0.001
    teammate_fp_plot.x_range = Range1d(x_min, x_max, bounds=(x_min, x_max))
    teammate_fp_plot.xaxis.ticker = FixedTicker(ticks=positions_source["x"])
    teammate_fp_plot.xaxis.major_label_overrides = {row["x"]: row["roundName"] for idx, row in
                                                    positions_source.iterrows()}
    teammate_fp_plot.xaxis.major_label_orientation = 0.8 * math.pi / 2

    return column([slider, teammate_fp_plot], sizing_mode="stretch_width"), source


def generate_stats_layout(positions_source, comparison_source, constructor_results, year_id, driver_id):
    """
    Year summary div, including:
    - WDC place
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
    :param positions_source: Positions source
    :param comparison_source: Comparison source (from teammate comparison line plot)
    :param constructor_results: Constructor results (from results.csv)
    :param year_id: Year
    :param driver_id: Driver ID
    :return: Stats layout
    """
    logging.info("Generating year driver stats layout")
    if positions_source.shape[0] == 0:
        return Div(text="")
    wdc_final_standing = positions_source["wdc_final_standing"].mode()
    if wdc_final_standing.shape[0] > 0:
        wdc_final_standing_str = int_to_ordinal(wdc_final_standing.values[0])
    else:
        wdc_final_standing_str = ""
    highest_race_finish_idx = positions_source["finish_position_int"].idxmin()
    if np.isnan(highest_race_finish_idx):
        highest_race_finish_str = ""
    else:
        highest_race_finish = positions_source.loc[highest_race_finish_idx, "finish_position_int"]
        round_name = positions_source.loc[highest_race_finish_idx, "roundName"]
        highest_race_finish_str = int_to_ordinal(highest_race_finish) + " at " + round_name
    num_races = positions_source.shape[0]
    num_races_str = str(num_races)
    points = positions_source["points"].max()
    if np.isnan(points):
        points_str = ""
    elif points <= 0:
        points_str = str(points) + " (0 pts/race)"
    else:
        points_str = str(points) + " (" + str(round(points / num_races, 1)) + " pts/race)"
    wins_slice = positions_source[positions_source["finish_position_int"] == 1]
    num_wins = wins_slice.shape[0]
    if num_wins == 0:
        wins_str = str(num_wins)
    else:
        wins_str = str(num_wins) + " (" + ", ".join(wins_slice["roundName"]) + ")"
        if len(wins_str) > 120:
            split = wins_str.split(" ")
            split.insert(int(len(split) / 2), "<br>    " + "".ljust(20))
            wins_str = " ".join(split)
    podiums_slice = positions_source[positions_source["finish_position_int"] <= 3]
    num_podiums = podiums_slice.shape[0]
    if num_podiums == 0:
        podiums_str = str(num_podiums)
    else:
        podiums_str = str(num_podiums) + " (" + ", ".join(podiums_slice["roundName"]) + ")"
        if len(podiums_str) > 120:
            split = podiums_str.split(" ")
            split.insert(int(len(split) / 2), "<br>    " + "".ljust(20))
            podiums_str = " ".join(split)
    teammate_dids = set(constructor_results["driverId"].unique()) - {driver_id}
    teammate_names = []
    for did in teammate_dids:
        teammate_names.append(get_driver_name(did, include_flag=False))
    teammate_str = ", ".join(teammate_names)
    constructor_cids = set(constructor_results["constructorId"].unique())
    constructor_names = []
    for cid in constructor_cids:
        constructor_names.append(get_constructor_name(cid, include_flag=True))
    constructors_str = ", ".join(constructor_names)
    mean_grid_pos = positions_source["grid"].replace("", np.nan).mean()
    if np.isnan(mean_grid_pos):
        mean_grid_pos_str = ""
    else:
        mean_grid_pos_str = str(round(mean_grid_pos, 1))
    mean_finish_pos = positions_source["finish_position_int"].mean()
    if np.isnan(mean_finish_pos):
        mean_finish_pos_str = ""
    else:
        mean_finish_pos_str = str(round(mean_finish_pos, 1))
    mean_teammate_gap_pos = (comparison_source["driver_fp"] - comparison_source["teammate_fp"]).mean()
    if np.isnan(mean_teammate_gap_pos):
        mean_teammate_gap_pos_str = ""
    else:
        mean_teammate_gap_pos_str = str(abs(round(mean_teammate_gap_pos, 1))) + " places " + \
                                    ("better" if mean_teammate_gap_pos < 0 else "worse") + " than teammate"

    # Construct the HTML
    header_template = """
    <h2 style="text-align: left;"><b>{}</b></h2>
    """
    template = """
    <pre><b>{}</b> {}<br></pre>
    """

    driver_name = get_driver_name(driver_id, include_flag=False)
    driver_stats = header_template.format(f"{driver_name}'s Stats for the {year_id} Season")
    driver_stats += template.format("WDC Final Pos.: ".ljust(20), wdc_final_standing_str)
    driver_stats += template.format("Num. Races: ".ljust(20), num_races_str)
    if num_wins == 0:
        driver_stats += template.format("Best Finish Pos.: ".ljust(20), highest_race_finish_str)
    driver_stats += template.format("Wins: ".ljust(20), wins_str)
    driver_stats += template.format("Podiums: ".ljust(20), podiums_str)
    driver_stats += template.format("Points: ".ljust(20), points_str)
    driver_stats += template.format("Constructor(s): ".ljust(20), constructors_str)
    driver_stats += template.format("Teammate(s): ".ljust(20), teammate_str)
    driver_stats += template.format("Avg. Grid Pos.: ".ljust(20), mean_grid_pos_str)
    driver_stats += template.format("Avg. Finish Pos.: ".ljust(20), mean_finish_pos_str)
    driver_stats += template.format("Avg. Gap to T.M.: ".ljust(20), mean_teammate_gap_pos_str)

    return Div(text=driver_stats)


def generate_error_layout(year_id, driver_id):
    """
    Generates an error layout in the event that the given driver didn't compete in the given season.
    :param year_id: Year
    :param driver_id: Driver ID
    :return: Div layout
    """
    logging.info("Generating error layout")
    driver_rids = results[results["driverId"] == driver_id]["raceId"].unique().tolist()
    driver_years = sorted(races.loc[driver_rids, "year"].unique().tolist())
    # Generate the text
    text = f"Unfortunately, {get_driver_name(driver_id, include_flag=False)} did not compete in the {year_id} " \
           f"season. He/she competed in the following seasons...<br>"
    text += "<ul>"
    for year in driver_years:
        text += f"<li>{str(year)}</li>"
    text += "</ul>"
    layout = Div(text=text)
    return layout


def is_valid_input(year_id, driver_id):
    """
    Returns whether the given input is a valid combination of year and driver. Used only for unit tests.
    :param year_id: Year
    :param driver_id: Driver ID
    :return: True if valid input, False otherwise
    """
    year_races = races[races["year"] == year_id]
    year_rids = sorted(year_races.index.values)
    year_results = results[results["raceId"].isin(year_rids)].sort_values(by="raceId")

    return driver_id in year_results["driverId"].unique()
