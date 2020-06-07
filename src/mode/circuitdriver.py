import logging
import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import Div, DatetimeTickFormatter, NumeralTickFormatter, LegendItem, Title, Range1d, Legend, \
    HoverTool, CrosshairTool
from bokeh.plotting import figure
from data_loading.data_loader import load_results, load_races, load_lap_times, load_drivers, load_fastest_lap_data, \
    load_status, load_circuits, load_driver_standings, load_constructor_standings
from mode import driver, yearconstructor, driverconstructor, yeardriver
from utils import get_circuit_name, get_driver_name, DATETIME_TICK_KWARGS, millis_to_str, get_constructor_name, \
    plot_image_url, int_to_ordinal, rounds_to_str, get_status_classification, \
    generate_plot_list_selector, generate_spacer_item, generate_div_item, PlotItem, COMMON_PLOT_DESCRIPTIONS

# Note, CD stands for circuit driver

results = load_results()
races = load_races()
lap_times = load_lap_times()
drivers = load_drivers()
fastest_lap_data = load_fastest_lap_data()
status = load_status()
circuits = load_circuits()
driver_standings = load_driver_standings()
constructor_standings = load_constructor_standings()


def get_layout(circuit_id=-1, driver_id=-1, download_image=True, **kwargs):
    # Grab some useful slices
    circuit_rids = races[races["circuitId"] == circuit_id].index.values
    driver_results = results[results["driverId"] == driver_id]
    cd_results = driver_results[driver_results["raceId"].isin(circuit_rids)]
    cd_rids = cd_results["raceId"]
    cd_races = races[races.index.isin(cd_rids)]

    logging.info(f"Generating layout for mode CIRCUITDRIVER in circuitdriver, circuit_id={circuit_id}, "
                 f"driver_id={driver_id}")

    # Detect invalid combos
    if cd_races.shape[0] == 0:
        return generate_error_layout(circuit_id, driver_id)

    # Grab some more slices
    cd_lap_times = lap_times[(lap_times["raceId"].isin(cd_rids)) & (lap_times["driverId"] == driver_id)]
    cd_years = cd_races["year"].unique()
    cd_years.sort()
    circuit_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"].isin(cd_rids)]
    cd_fastest_lap_data = circuit_fastest_lap_data[circuit_fastest_lap_data["driver_id"] == driver_id]
    driver_driver_standings = driver_standings[driver_standings["driverId"] == driver_id]
    circuit_results = results[results["raceId"].isin(circuit_rids)]
    constructor_results_idxs = []
    for idx, results_row in cd_results.iterrows():
        constructor_id = results_row["constructorId"]
        rid = results_row["raceId"]
        results_slice = circuit_results[(circuit_results["constructorId"] == constructor_id) &
                                        (circuit_results["raceId"] == rid)]
        constructor_results_idxs.extend(results_slice.index.values.tolist())
    constructor_results = circuit_results.loc[constructor_results_idxs]

    positions_plot, positions_source = generate_positions_plot(cd_years, driver_driver_standings, driver_results,
                                                               cd_fastest_lap_data, cd_races, driver_id)
    positions_plot = PlotItem(positions_plot, [], COMMON_PLOT_DESCRIPTIONS["generate_positions_plot"])

    win_plot = PlotItem(generate_win_plot, [positions_source], COMMON_PLOT_DESCRIPTIONS["generate_win_plot"])

    lap_time_dist = PlotItem(generate_lap_time_plot, [cd_lap_times, cd_rids, circuit_id, driver_id],
                             COMMON_PLOT_DESCRIPTIONS["generate_times_plot"])

    spvfp_scatter = PlotItem(generate_spvfp_scatter, [cd_results, cd_races, driver_driver_standings],
                             COMMON_PLOT_DESCRIPTIONS["generate_spvfp_scatter"])

    mltr_fp_scatter = PlotItem(generate_mltr_fp_scatter, [cd_results, cd_races, driver_driver_standings],
                               COMMON_PLOT_DESCRIPTIONS["generate_mltr_fp_scatter"])

    finish_pos_dist = PlotItem(generate_finishing_position_bar_plot, [cd_results],
                               COMMON_PLOT_DESCRIPTIONS["generate_finishing_position_bar_plot"])

    teammate_comparison_line_plot = PlotItem(generate_teammate_comparison_line_plot, [positions_source,
                                                                                      constructor_results, cd_results,
                                                                                      driver_id],
                                             COMMON_PLOT_DESCRIPTIONS["generate_teammate_comparison_line_plot"])

    description = u"Results Table \u2014 results for every year this driver was at this circuit"
    results_table = PlotItem(generate_results_table, [cd_results, cd_fastest_lap_data, circuit_results,
                                                      circuit_fastest_lap_data], description)

    description = u"Various statistics on this driver at this circuit"
    stats_div = PlotItem(generate_stats_layout, [cd_years, cd_races, cd_results, cd_fastest_lap_data, positions_source,
                                                 circuit_id, driver_id], description)

    if download_image:
        # Track image
        circuit_row = circuits.loc[circuit_id]
        image_url = str(circuit_row["imgUrl"])
        image_view = plot_image_url(image_url)
        disclaimer = Div(text="The image is of the current configuration of the track.")
        image_view = column([image_view, disclaimer], sizing_mode="stretch_both")
    else:
        image_view = Div()
    image_view = PlotItem(image_view, [], "", listed=False)

    header = generate_div_item(f"<h2><b>{get_driver_name(driver_id)} at {get_circuit_name(circuit_id)}</b></h2><br>")

    middle_spacer = generate_spacer_item()
    group = generate_plot_list_selector([
        [header],
        [positions_plot], [middle_spacer],
        [win_plot], [middle_spacer],
        [lap_time_dist], [middle_spacer],
        [finish_pos_dist], [middle_spacer],
        [spvfp_scatter, mltr_fp_scatter], [middle_spacer],
        [teammate_comparison_line_plot],
        [image_view],
        [results_table],
        [stats_div]
    ])

    logging.info("Finished generating layout for mode CIRCUITDRIVER")

    return group


def generate_lap_time_plot(cd_lap_times, cd_rids, circuit_id, driver_id, constructor_id=None):
    """
    Plot lap time distribution of the driver at this circuit along with the lap time distribution of all drivers at this
    circuit during the time period to show how fast and consistent he is.
    :param cd_lap_times: Circuit driver lap times
    :param cd_rids: Circuit driver race IDs
    :param circuit_id: Circuit ID
    :param driver_id: Driver ID, can be set to None if using constructor mode
    :param constructor_id: Constructor ID, set to None if not using constructor mode
    :return: Lap time plot layout
    """
    logging.info("Generating lap time distribution plot")
    # Collect data on everyone during these years
    all_times = lap_times[lap_times["raceId"].isin(cd_rids)]
    millis_range_min = all_times["milliseconds"].mean() - 1 * all_times["milliseconds"].std()
    millis_range_max = all_times["milliseconds"].mean() + 2 * all_times["milliseconds"].std()

    cd_lap_times = cd_lap_times[(cd_lap_times["milliseconds"] > millis_range_min) &
                                (cd_lap_times["milliseconds"] < millis_range_max)]
    if cd_lap_times.shape[0] == 0:
        return Div(text="Unfortunately, we do not yet have lap time data on this driver at this circuit.")
    all_times = all_times[(all_times["milliseconds"] > millis_range_min) &
                          (all_times["milliseconds"] < millis_range_max)]

    cd_hist, cd_edges = np.histogram(cd_lap_times["milliseconds"], bins=50)
    all_hist, all_edges = np.histogram(all_times["milliseconds"], bins=50)
    cd_hist = cd_hist / cd_lap_times.shape[0]
    all_hist = all_hist / all_times.shape[0]

    all_pdf_source = pd.DataFrame(columns=["x", "pdf"])
    cd_pdf_source = pd.DataFrame(columns=["x", "pdf"])
    for i in range(0, all_edges.shape[0] - 1):
        x = 0.5 * (all_edges[i] + all_edges[i + 1])
        pdf = all_hist[i]
        all_pdf_source = all_pdf_source.append({"x": x, "pdf": pdf}, ignore_index=True)
    for i in range(0, cd_edges.shape[0] - 1):
        x = 0.5 * (cd_edges[i] + cd_edges[i + 1])
        pdf = cd_hist[i]
        cd_pdf_source = cd_pdf_source.append({"x": x, "pdf": pdf}, ignore_index=True)
    all_pdf_source["lap_time_str"] = all_pdf_source["x"].apply(millis_to_str)
    cd_pdf_source["lap_time_str"] = all_pdf_source["x"].apply(millis_to_str)
    all_pdf_source["pct_str"] = all_pdf_source["pdf"].apply(lambda pdf: str(round(100 * pdf, 1)) + "%")
    cd_pdf_source["pct_str"] = cd_pdf_source["pdf"].apply(lambda pdf: str(round(100 * pdf, 1)) + "%")

    if constructor_id:
        name = get_constructor_name(constructor_id, include_flag=False)
    else:
        name = get_driver_name(driver_id, include_flag=False, just_last=True)
    circuit_name = get_circuit_name(circuit_id, include_flag=False)
    title = u"Lap Time Distribution \u2014 " + name + "'s lap times at " + circuit_name + \
            " vs the rest of the field during their years"
    max_y = 0.02 + max(cd_pdf_source["pdf"].max(), all_pdf_source["pdf"].max())
    min_x = all_edges[0] - 500
    max_x = all_edges[-1] + 500
    time_dist = figure(title=title,
                       y_axis_label="% Occurrence",
                       x_axis_label="Lap Time",
                       y_range=Range1d(0, max_y, bounds=(0, max_y)),
                       x_range=Range1d(min_x, max_x, bounds=(min_x, max_x + 3)),
                       tools="pan,box_zoom,wheel_zoom,reset,save"
                       )
    subtitle = "Only lap times within 2 standard deviations of the mean are shown, means marked with horizontal line"
    time_dist.add_layout(Title(text=subtitle, text_font_style="italic"), "above")
    time_dist.xaxis.formatter = DatetimeTickFormatter(**DATETIME_TICK_KWARGS)
    time_dist.yaxis.formatter = NumeralTickFormatter(format="0.0%")

    cd_quad = time_dist.quad(top=cd_hist, bottom=0, left=cd_edges[:-1], right=cd_edges[1:],
                             fill_color="orange", line_alpha=0, alpha=0.1, muted_alpha=0)

    line_kwargs = dict(
        x="x",
        y="pdf",
        line_alpha=0.9,
        line_width=2,
        muted_line_alpha=0.05
    )
    cd_pdf_line = time_dist.line(source=cd_pdf_source, color="orange", **line_kwargs)
    all_pdf_line = time_dist.line(source=all_pdf_source, color="white", **line_kwargs)

    # Mark means
    line_kwargs = dict(
        y=[-100, 100],
        line_alpha=0.9,
        line_width=2,
        muted_alpha=0.05
    )
    all_mean = all_times["milliseconds"].mean()
    cd_mean = cd_lap_times["milliseconds"].mean()
    cd_mean_line = time_dist.line(x=[cd_mean] * 2, line_color="orange", **line_kwargs)
    all_mean_line = time_dist.line(x=[all_mean] * 2, line_color="white", **line_kwargs)

    # Legend
    legend = [LegendItem(label=f"{name}'s Dist.", renderers=[cd_pdf_line, cd_quad, cd_mean_line]),
              LegendItem(label="All Drivers Dist.", renderers=[all_pdf_line, all_mean_line])]

    legend = Legend(items=legend, location="top_right", glyph_height=15, spacing=2, inactive_fill_color="gray")
    time_dist.add_layout(legend, "right")
    time_dist.legend.click_policy = "mute"
    time_dist.legend.label_text_font_size = "12pt"

    # Hover tooltip
    time_dist.add_tools(HoverTool(show_arrow=False, renderers=[all_pdf_line, cd_pdf_line], tooltips=[
        ("Lap Time", "@lap_time_str"),
        ("Percent of Laps", "@pct_str")
    ]))

    # Crosshair tooltip
    time_dist.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

    return time_dist


def generate_positions_plot(cd_years, driver_driver_standings, driver_results, cd_fastest_lap_data, cd_races,
                            driver_id):
    """
    Plot quali, finishing, and mean finish position and average lap time rank vs time to show improvement and on the
    same plot but different axis, plot average race times to show car+driver improvement.
    :param cd_years: CD years
    :param driver_driver_standings: Driver driver standings
    :param driver_results: Driver results
    :param cd_fastest_lap_data: CD fastest lap data
    muted by default
    :param cd_races: CD races
    :param driver_id: Driver ID
    :return: Position plot layout
    """
    return driver.generate_positions_plot(cd_years, driver_driver_standings, driver_results, cd_fastest_lap_data,
                                          driver_id, races_sublist=cd_races, smoothing_alpha=0.65,
                                          show_mean_finish_pos=True, include_lap_times=True)


def generate_results_table(cd_results, cd_fastest_lap_data, circuit_results, circuit_fastest_lap_data):
    """
    Table of all results for this driver at this circuit.
    :param cd_results: CD results
    :param cd_fastest_lap_data: CD fastest lap data
    :param circuit_results: Circuit results
    :param circuit_fastest_lap_data: Circuit fastest lap data
    :return: Results table layout, source
    """
    return yearconstructor.generate_results_table(cd_results, cd_fastest_lap_data, circuit_results,
                                                  circuit_fastest_lap_data, year_only=True, include_driver_name=False,
                                                  include_constructor_name=True)


def generate_win_plot(positions_source):
    """
    Plot number of races, number of wins, number of podiums, and maybe also number of top 5 or 6 finishes vs time all on
    one graph (two axes) for this circuit
    :param positions_source: Positions source
    :return: Win plot
    """
    # TODO add back top-n support when it is implemented in the driver method
    return driver.generate_win_plot(positions_source)
    # logging.info("Generating win plot")
    # win_source = pd.DataFrame(columns=["year",
    #                                    "n_races",
    #                                    "win_pct", "wins", "win_pct_str",
    #                                    "podium_pct", "podiums", "podium_pct_str",
    #                                    "top6_pct", "top6", "top6_pct_str",
    #                                    "constructor_name"])
    # wins = 0
    # podiums = 0
    # top6 = 0
    # n_races = 0
    # for idx, row in positions_source.sort_values(by="year").iterrows():
    #     year = row["year"]
    #     pos = row["finish_position_int"]
    #     if not np.isnan(pos):
    #         wins += 1 if pos == 1 else 0
    #         podiums += 1 if 3 >= pos > 0 else 0
    #         top6 += 1 if 6 >= pos > 0 else 0
    #     n_races += 1
    #     win_pct = wins / n_races
    #     podium_pct = podiums / n_races
    #     top6_pct = top6 / n_races
    #     win_source = win_source.append({
    #         "year": year,
    #         "n_races": n_races,
    #         "wins": wins,
    #         "win_pct": win_pct,
    #         "podiums": podiums,
    #         "podium_pct": podium_pct,
    #         "top6": top6,
    #         "top6_pct": top6_pct,
    #         "constructor_name": row["constructor_name"],
    #         "win_pct_str": str(round(100 * win_pct, 1)) + "%",
    #         "podium_pct_str": str(round(100 * podium_pct, 1)) + "%",
    #         "top6_pct_str": str(round(100 * top6_pct, 1)) + "%"
    #     }, ignore_index=True)
    #
    # min_year = positions_source["year"].min()
    # max_year = positions_source["year"].max()
    # win_plot = figure(
    #     title=u"Wins, Podiums, and Top 6 finishes",
    #     y_axis_label="",
    #     x_axis_label="Year",
    #     x_range=Range1d(min_year, max_year, bounds=(min_year - 1, max_year + 1)),
    #     tools="pan,xbox_zoom,xwheel_zoom,reset,box_zoom,wheel_zoom,save",
    #     y_range=Range1d(0, win_source["top6"].max() + 1, bounds=(-3, 25))
    # )
    # win_plot.xaxis.ticker = FixedTicker(ticks=np.arange(min_year - 1, max_year + 2))
    #
    # max_top6_pct = win_source["top6_pct"].max()
    # max_top6 = win_source["top6"].max()
    # if max_top6 == 0:
    #     k = 1
    # else:
    #     k = max_top6 / max_top6_pct
    # win_source["top6_pct_scaled"] = k * win_source["top6_pct"]
    # win_source["podium_pct_scaled"] = k * win_source["podium_pct"]
    # win_source["win_pct_scaled"] = k * win_source["win_pct"]
    #
    # # Other y axis
    # max_y = win_plot.y_range.end
    # y_range = Range1d(start=0, end=max_y / win_source["n_races"].max(), bounds=(-0.02, 1000))
    # win_plot.extra_y_ranges = {"percent_range": y_range}
    # axis = LinearAxis(y_range_name="percent_range")
    # axis.formatter = NumeralTickFormatter(format="0.0%")
    # win_plot.add_layout(axis, "right")
    #
    # kwargs = {
    #     "x": "year",
    #     "line_width": 2,
    #     "line_alpha": 0.7,
    #     "source": win_source,
    #     "muted_alpha": 0.05
    # }
    # races_line = win_plot.line(y="n_races", color="white", **kwargs)
    # wins_line = win_plot.line(y="wins", color="green", **kwargs)
    # win_pct_line = win_plot.line(y="win_pct_scaled", color="green", line_dash="dashed", **kwargs)
    # podiums_line = win_plot.line(y="podiums", color="yellow", **kwargs)
    # podium_pct_line = win_plot.line(y="podium_pct_scaled", color="yellow", line_dash="dashed", **kwargs)
    # top6_line = win_plot.line(y="top6", color="red", **kwargs)
    # top6_pct_line = win_plot.line(y="top6_pct_scaled", color="red", line_dash="dashed", **kwargs)
    #
    # legend = [LegendItem(label="Number of Races", renderers=[races_line]),
    #           LegendItem(label="Number of Wins", renderers=[wins_line]),
    #           LegendItem(label="Win Percentage", renderers=[win_pct_line]),
    #           LegendItem(label="Number of Podiums", renderers=[podiums_line]),
    #           LegendItem(label="Podium Percentage", renderers=[podium_pct_line]),
    #           LegendItem(label="Number of Top 6s", renderers=[top6_line]),
    #           LegendItem(label="Top 6 Percentage", renderers=[top6_pct_line])]
    #
    # legend = Legend(items=legend, location="top_right", glyph_height=15, spacing=2, inactive_fill_color="gray")
    # win_plot.add_layout(legend, "right")
    # win_plot.legend.click_policy = "mute"
    # win_plot.legend.label_text_font_size = "12pt"
    #
    # # Hover tooltip
    # win_plot.add_tools(HoverTool(show_arrow=False, tooltips=[
    #     ("Year", "@year"),
    #     ("Number of Races", "@n_races"),
    #     ("Number of Wins", "@wins (@win_pct_str)"),
    #     ("Number of Podiums", "@podiums (@podium_pct_str)"),
    #     ("Number of Top 6 Finishes", "@top6 (@top6_pct_str)"),
    #     ("Constructor", "@constructor_name")
    # ]))
    #
    # # Crosshair tooltip
    # win_plot.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))
    #
    # return win_plot


def generate_spvfp_scatter(cd_results, cd_races, driver_driver_standings):
    """
    Plot a scatter of quali position vs finish position and draw the y=x line
    :param cd_results: CD results
    :param cd_races: CD races
    :param driver_driver_standings: Driver driver standings
    :return: Start pos. vs finish pos. scatter layout
    """
    return driver.generate_spvfp_scatter(cd_results, cd_races, driver_driver_standings, include_year_labels=True)


def generate_mltr_fp_scatter(cd_results, cd_races, driver_driver_standings):
    """
    Plot scatter of mean lap time rank (x) vs finish position (y) to get a sense of what years the driver out-drove the
    car
    :param cd_results: CD results
    :param cd_races: CD races
    :param driver_driver_standings: Driver driver standings
    :return: Mean lap time rank vs finish position scatter plot layout
    """
    return driver.generate_mltr_fp_scatter(cd_results, cd_races, driver_driver_standings, include_year_labels=True)


def generate_teammate_comparison_line_plot(positions_source, constructor_results, cd_results, driver_id):
    """
    Driver finish pos and teammate finish pos vs time.
    :param positions_source: Positions source
    :param constructor_results: Constructor results
    :param cd_results: CD results
    :param driver_id: Driver ID
    :return: Teammate comparison line plot layout
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

    yeardriver.mark_teammate_team_changes(cd_results, positions_source, driver_id, teammate_fp_plot, x_offset=0.2)

    return column([slider, teammate_fp_plot], sizing_mode="stretch_width")


def generate_finishing_position_bar_plot(cd_results):
    """
    Bar plot of finishing positions at this circuit.
    :param cd_results: CD results
    :return: Bar plot layout
    """
    return driver.generate_finishing_position_bar_plot(cd_results)


def generate_stats_layout(cd_years, cd_races, cd_results, cd_fastest_lap_data, positions_source, circuit_id, driver_id,
                          constructor_id=None):
    """
    Stats div including:
    - Years
    - Num. races
    - Num. wins
    - Num. podiums
    - Best results
    - Average start position
    - Average finish position
    - Average lap time
    - Fastest lap time
    - Num mechanical DNFs and mechanical DNF rate
    - Num crash DNFs and crash DNF rate
    :param cd_years: CD years
    :param cd_races: CD races
    :param cd_results: CD results
    :param cd_fastest_lap_data: CD fastest lap data
    :param positions_source: Positions source
    :param driver_id: Driver ID
    :param circuit_id: Circuit ID
    :param constructor_id: If set to anything but None, will do constructor mode
    :return: Stats div layout
    """
    logging.info("Generating stats div")
    num_races = cd_results.shape[0]
    if num_races == 0:
        return Div()
    win_results = cd_results[cd_results["positionOrder"] == 1]
    num_wins = win_results.shape[0]
    if num_wins > 0:
        rids = win_results["raceId"]
        years = sorted(cd_races.loc[rids.values, "year"].astype(str).values.tolist(), reverse=True)
        num_wins = str(num_wins) + " (" + ", ".join(years) + ")"
    else:
        num_wins = str(num_wins)
    podium_results = cd_results[cd_results["positionOrder"] <= 3]
    num_podiums = podium_results.shape[0]
    if num_podiums > 0:
        rids = podium_results["raceId"]
        years = list(set(cd_races.loc[rids.values, "year"].values.tolist()))
        years = rounds_to_str(years)
        num_podiums_str = str(num_podiums) + " (" + years + ")"
        if len(num_podiums_str) > 120:
            split = num_podiums_str.split(" ")
            split.insert(int(len(split) / 2), "<br>      " + "".ljust(20))
            num_podiums_str = " ".join(split)
    else:
        num_podiums_str = str(num_podiums)
    best_result = None
    if num_wins == 0:
        idxmin = cd_results["positionOrder"].idxmin()
        if not np.isnan(idxmin):
            rid = cd_results.loc[idxmin, "raceId"]
            year = cd_races.loc[rid, "year"]
            best_result = int_to_ordinal(int(cd_results.loc[idxmin, "positionOrder"])) + f" ({year})"
    mean_sp = round(cd_results["grid"].mean(), 1)
    mean_fp = round(cd_results["positionOrder"].mean(), 1)

    avg_lap_time = cd_fastest_lap_data["avg_lap_time_millis"].mean()
    fastest_lap_time = cd_fastest_lap_data["fastest_lap_time_millis"].min()

    classifications = cd_results["statusId"].apply(get_status_classification)
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

    if positions_source.shape[0] > 0:
        avg_finish_pos_overall = positions_source["avg_finish_pos"].mean()
        avg_finish_pos_here = positions_source["finish_position_int"].mean()
        diff = avg_finish_pos_here - avg_finish_pos_overall
        avg_finish_pos_overall = round(avg_finish_pos_overall, 1)
        avg_finish_pos_here = round(avg_finish_pos_here, 1)
        w = "higher" if diff < 0 else "lower"
        finish_pos_diff_str = f"Finished on average {round(abs(diff), 1)} place(s) {w} than average " \
                              f"(pos. {avg_finish_pos_here} here vs pos. {avg_finish_pos_overall} average overall)"
    else:
        finish_pos_diff_str = ""

    header_template = """
    <h2 style="text-align: center;"><b>{}</b></h2>
    """

    template = """
    <pre><b>{}</b> {}<br></pre>
    """

    if constructor_id:
        name = get_constructor_name(constructor_id, include_flag=False)
    else:
        name = get_driver_name(driver_id, include_flag=False, just_last=True)
    cd_stats = header_template.format(f"{name} at {get_circuit_name(circuit_id, include_flag=False)} Stats")
    cd_stats += template.format("Years: ".ljust(22), rounds_to_str(cd_years))
    cd_stats += template.format("Num Races: ".ljust(22), str(num_races))
    cd_stats += template.format("Num Wins: ".ljust(22), str(num_wins))
    cd_stats += template.format("Num Podiums: ".ljust(22), str(num_podiums_str))
    if best_result:
        cd_stats += template.format("Best Result: ".ljust(22), str(best_result))
    cd_stats += template.format("Avg. Start Pos.: ".ljust(22), mean_sp)
    cd_stats += template.format("Avg. Finish Pos.: ".ljust(22), mean_fp)

    if not np.isnan(avg_lap_time):
        cd_stats += template.format("Avg. Lap Time: ".ljust(22), millis_to_str(avg_lap_time))
        cd_stats += template.format("Fastest Lap Time: ".ljust(22), millis_to_str(fastest_lap_time))
    cd_stats += template.format("Num. Mechanical DNFs: ".ljust(22), mechanical_dnfs_str)
    cd_stats += template.format("Num. Crash DNFs: ".ljust(22), crash_dnfs_str)
    cd_stats += template.format("Num Finishes".ljust(22), finishes_str)
    if positions_source.shape[0] > 0:
        cd_stats += template.format("Compared to Average: ".ljust(22), finish_pos_diff_str)

    return Div(text=cd_stats)


def generate_error_layout(circuit_id, driver_id):
    """
    Generates an error layout in the event that the given driver never competed at the given circuit.
    :param circuit_id: Circuit ID
    :param driver_id: Driver ID
    :return: Div layout
    """
    logging.info("Generating error layout")
    circuit_name = get_circuit_name(circuit_id, include_flag=False)
    driver_name = get_driver_name(driver_id, include_flag=False)
    circuit_rids = races[races["circuitId"] == circuit_id].index.values
    drivers_at_that_circuit = results[results["raceId"].isin(circuit_rids)]["driverId"].unique()
    rids_for_this_driver = results[results["driverId"] == driver_id]["raceId"]
    circuits_for_this_driver = races[races.index.isin(rids_for_this_driver)]["circuitId"].unique()

    # Generate the text
    text = f"Unfortunately, {driver_name} never competed at {circuit_name}. Here are some other options:<br>"
    text += f"{driver_name} competed at the following circuits..."
    text += "<ul>"
    for cid in circuits_for_this_driver:
        text += f"<li>{get_circuit_name(cid)}</li>"
    text += "</ul>"
    text += f"The {circuit_name} hosted the following drivers..."
    text += "<ul>"
    for did in drivers_at_that_circuit:
        text += f"<li>{get_driver_name(did)}</li>"
    text += "</ul><br>"

    layout = Div(text=text)

    return layout


def is_valid_input(circuit_id, driver_id):
    """
    Returns whether the given input is a valid combination of circuit and driver. Used only for unit tests.
    :param circuit_id: Circuit ID
    :param driver_id: Driver ID
    :return: True if valid input, False otherwise
    """
    circuit_rids = races[races["circuitId"] == circuit_id].index.values
    cd_results = results[(results["raceId"].isin(circuit_rids)) & (results["driverId"] == driver_id)]
    cd_races = races[races.index.isin(cd_results["raceId"])]
    return cd_races.shape[0] > 0
