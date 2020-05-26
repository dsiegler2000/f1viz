import logging
import math
from bokeh.layouts import column, row
from bokeh.models import Div, Spacer, Range1d, FixedTicker
from pandas import Series
from data_loading.data_loader import load_results, load_races, load_driver_standings, load_constructor_standings, \
    load_fastest_lap_data
from mode import year, driver, driverconstructor, yeardriver
import numpy as np
from utils import get_driver_name, get_constructor_name, PLOT_BACKGROUND_COLOR

# Note, ydc=yeardriverconstructor and yd=yeardriver

results = load_results()
races = load_races()
driver_standings = load_driver_standings()
constructor_standings = load_constructor_standings()
fastest_lap_data = load_fastest_lap_data()


def get_layout(year_id=-1, driver_id=-1, constructor_id=-1, **kwargs):
    # Generate slices
    year_races = races[races["year"] == year_id]
    year_rids = year_races.index.values
    year_results = results[results["raceId"].isin(year_rids)]
    yd_results = year_results[year_results["driverId"] == driver_id]

    logging.info(f"Generating layout for mode YEARDRIVERCONSTRUCTOR in yeardriverconstructor, year_id={year_id}, "
                 f"driver_id={driver_id}, constructor_id={constructor_id}")

    # Check if valid
    if yd_results.shape[0] == 0:
        return generate_error_layout(year_id, driver_id, constructor_id)

    # Generate more slices
    yd_races = year_races.loc[yd_results["raceId"].values]
    year_driver_standings = driver_standings[driver_standings["raceId"].isin(year_rids)]
    yd_driver_standings = year_driver_standings[year_driver_standings["driverId"] == driver_id]
    year_constructor_standings = constructor_standings[constructor_standings["raceId"].isin(year_rids)]
    year_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"].isin(year_rids)]
    yd_fastest_lap_data = year_fastest_lap_data[year_fastest_lap_data["driver_id"] == driver_id]
    constructor_results = year_results[year_results["constructorId"] == constructor_id]

    # WDC plot
    wdc_plot = generate_wdc_plot(year_races, year_driver_standings, year_results, driver_id, consider_window=2)

    # WCC plot
    wcc_plot = generate_wcc_plot(year_races, year_constructor_standings, year_results, constructor_id)

    # Positions plot
    positions_plot, positions_source = generate_positions_plot(yd_results, yd_driver_standings, yd_fastest_lap_data,
                                                               driver_id, year_id)

    # Start pos vs finish pos scatter
    spvfp_scatter = generate_spvfp_scatter(yd_results, yd_races, yd_driver_standings)

    # Mean lap time rank vs finish pos scatter
    mltr_fp_scatter = generate_mltr_fp_scatter(yd_results, yd_races, yd_driver_standings)

    # Win plot
    win_plot = generate_win_plot(positions_source, driver_id)

    # Finishing position bar plot
    position_dist = generate_finishing_position_bar_plot(yd_results)

    # Teammate finish pos vs finish pos scatter
    teammatefp_fp_scatter = generate_teammatefp_fp_scatter(positions_source, constructor_results, driver_id)

    # Teammate diff plot
    teammate_diff_plot, explanation_div, teammate_diff_source = generate_teammate_diff_comparison_scatter(
        positions_source, constructor_results, driver_id)

    # Teammate comparison line plot
    teammate_comparison_line_plot, comparison_source = generate_teammate_comparison_line_plot(positions_source,
                                                                                              constructor_results,
                                                                                              driver_id)

    # Results table
    results_table = generate_results_table(yd_results, yd_fastest_lap_data, year_results, year_fastest_lap_data,
                                           driver_id)

    # Stats layout
    stats_layout = generate_stats_layout(positions_source, comparison_source, constructor_results, year_id, driver_id)

    driver_name = get_driver_name(driver_id)
    constructor_name = get_constructor_name(constructor_id)
    header = Div(text=f"<h2><b>What did {driver_name}'s {year_id} season with {constructor_name} look like?</b></h2>")

    middle_spacer = Spacer(width=5, background=PLOT_BACKGROUND_COLOR)
    layout = column([header,
                     wdc_plot, middle_spacer,
                     wcc_plot, middle_spacer,
                     positions_plot, middle_spacer,
                     win_plot,
                     row([spvfp_scatter, mltr_fp_scatter], sizing_mode="stretch_width"), middle_spacer,
                     position_dist, middle_spacer,
                     row([teammatefp_fp_scatter, teammate_diff_plot], sizing_mode="stretch_width"),
                     explanation_div, middle_spacer,
                     teammate_comparison_line_plot,
                     results_table,
                     stats_layout],
                    sizing_mode="stretch_width")

    logging.info("Finished generating layout for mode YEARDRIVERCONSTRUCTOR")

    return layout


def generate_wdc_plot(year_races, year_driver_standings, year_results, driver_id, consider_window=2):
    """
    WDC plot, uses year.generate_wdc_plot.
    :param year_races: Year races
    :param year_driver_standings: Year driver standings
    :param year_results: Year results
    :param driver_id: Driver ID
    :param consider_window: Consider window
    :return: WDC plot layout
    """
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
        return year.generate_wdc_plot(year_driver_standings, year_results, highlight_did=[driver_id],
                                      muted_dids=muted_dids)
    else:
        driver_name = get_driver_name(driver_id, include_flag=False)
        return Div(text=f"Unfortunately, we have encountered an error or {driver_name} was never officially classified "
                        f"in this season.")


def generate_wcc_plot(year_races, year_constructor_standings, year_results, constructor_id, consider_window=2):
    """
    WCC plot, uses year.generate_wcc_plot.
    :param year_races: Year races
    :param year_constructor_standings: Year constructor standings
    :param year_results: Year results
    :param constructor_id: Constructor ID
    :param consider_window: Consider windows
    :return: WCC plot layout
    """
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
        return year.generate_wcc_plot(year_constructor_standings, year_results,
                                      highlight_cid=[constructor_id], muted_cids=muted_cids)
    else:
        constructor_name = get_constructor_name(constructor_id, include_flag=False)
        return Div(text=f"Unfortunately, we have encountered an error or {constructor_name} was never officially "
                        f"classified in this season.")


def generate_positions_plot(yd_results, yd_driver_standings, yd_fastest_lap_data, driver_id, year_id):
    """
    Plot WDC position (both rounds and full season), quali, fastest lap, and finishing position rank vs time all on the
    same graph along with smoothed versions of the quali, fastest lap, and finish position ranks.
    Uses driver.generate_positions_plot
    :param yd_results: YD results
    :param yd_driver_standings: YD driver standings
    :param yd_fastest_lap_data: YD fastest lap data
    :param driver_id: Driver ID
    :param year_id: Year ID
    :return: Positions plot, positions source
    """
    driver_years = np.array([year_id])
    driver_name = get_driver_name(driver_id)
    kwargs = dict(
        title=driver_name + " in " + str(year_id),
        smoothing_muted=True,
        smoothing_alpha=0.5
    )
    return driver.generate_positions_plot(driver_years, yd_driver_standings, yd_results, yd_fastest_lap_data, driver_id,
                                          **kwargs)


def generate_spvfp_scatter(yd_results, yd_races, yd_driver_standings):
    """
    Start pos vs finish pos scatter, uses driver.generate_spvfp_scatter.
    :param yd_results: YD results
    :param yd_races: YD races
    :param yd_driver_standings: YD driver standings
    :return: Start pos vs finish pos
    """
    return driver.generate_spvfp_scatter(yd_results, yd_races, yd_driver_standings, include_race_labels=True)


def generate_mltr_fp_scatter(yd_results, yd_races, yd_driver_standings):
    """
    Mean lap time rank vs finish pos scatter, uses driver.generate_mltr_fp_scatter.
    :param yd_results: YD results
    :param yd_races: YD races
    :param yd_driver_standings: YD driver standings
    :return: Mean lap time rank vs finish pos scatter
    """
    return driver.generate_mltr_fp_scatter(yd_results, yd_races, yd_driver_standings, include_race_labels=True)


def generate_results_table(yd_results, yd_fastest_lap_data, year_results, year_fastest_lap_data, driver_id):
    """
    Table of all results this season, uses yeardriver.generate_results_table
    :param yd_results: YD results
    :param yd_fastest_lap_data: YD fastest lap data
    :param year_results: Year results
    :param year_fastest_lap_data: Year fastest lap data
    :param driver_id: Driver ID
    :return: Results table
    """
    return yeardriver.generate_results_table(yd_results, yd_fastest_lap_data, year_results, year_fastest_lap_data,
                                             driver_id)


def generate_win_plot(positions_source, driver_id):
    """
    Plot number of races, number of wins, number of podiums, win percent, and podium percent on one plot.
    Uses driver.generate_win_plot.
    :param positions_source: Positions source
    :param driver_id: Driver ID
    :return: Win plot layout
    """
    return driver.generate_win_plot(positions_source, driver_id)


def generate_finishing_position_bar_plot(yd_results):
    """
    Plot race finishing position bar chart. Uses driver.generate_finishing_position_bar_plot.
    :param yd_results: YD results
    :return: Finishing position bar plot layout
    """
    return driver.generate_finishing_position_bar_plot(yd_results)


def generate_stats_layout(positions_source, comparison_source, constructor_results, year_id, driver_id):
    """
    mean grid position, mean finish position, mean gap to teammate (in positions and time), wins, and podiums
    Stats div including:
    - WDC position
    - WCC position
    - Num races
    - Num wins
    - Num podiums
    - Num mechanical DNFs
    - Num crash DNFs
    - Points scored and points per race
    - Mean finish position
    - Mean gap to teammate in positions
    - Mean gap to teammate in time
    - Num finishes
    Use yeardriver.generate_stats_layout.
    :param positions_source: Positions source
    :param comparison_source: Comparison source (from teammate comparison line plot)
    :param constructor_results: Constructor results (from results.csv)
    :param year_id: Year
    :param driver_id: Driver ID
    :return: Stats layout
    """
    # TODO add the constructor-specific stuff
    return yeardriver.generate_stats_layout(positions_source, comparison_source, constructor_results,
                                            year_id, driver_id)


def generate_teammatefp_fp_scatter(positions_source, constructor_results, driver_id):
    """
    Scatter plot of driver finish position vs teammate finish position to show if he is beating his teammate.
    Uses driverconstructor.generate_teammatefp_fp_scatter.
    :param positions_source: Positions source
    :param constructor_results: Constructor results (from results.csv)
    :param driver_id: Driver ID
    :return: Teammate finish position vs driver finish position scatter
    """
    kwargs = dict(include_year_labels=False,
                  include_race_labels=True)
    return driverconstructor.generate_teammatefp_fp_scatter(positions_source, constructor_results, driver_id, **kwargs)


def generate_teammate_comparison_line_plot(positions_source, constructor_results, driver_id):
    """
    Plot driver finish pos and teammate finish pos vs time.
    Uses driverconstructor.generate_teammate_comparison_line_plot.
    :param positions_source: Positions source
    :param constructor_results: Constructor results (from results.csv)
    :param driver_id: Driver ID
    :return: Teammate comparison line plot layout, comparison source
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

    driverconstructor.mark_teammate_changes(positions_source, constructor_results, driver_id, teammate_fp_plot)

    # x axis override
    x_min = positions_source["x"].min() - 0.001
    x_max = positions_source["x"].max() + 0.001
    teammate_fp_plot.x_range = Range1d(x_min, x_max, bounds=(x_min, x_max))
    teammate_fp_plot.xaxis.ticker = FixedTicker(ticks=positions_source["x"])
    teammate_fp_plot.xaxis.major_label_overrides = {row["x"]: row["roundName"] for idx, row in
                                                    positions_source.iterrows()}
    teammate_fp_plot.xaxis.major_label_orientation = 0.8 * math.pi / 2

    return column([slider, teammate_fp_plot], sizing_mode="stretch_width"), source


def generate_teammate_diff_comparison_scatter(positions_source, constructor_results, driver_id):
    """
    Try a scatter plot where the x axis is mean lap time rank difference from teammate (his avg time rank - teammate's),
    y axis is position difference from teammate
    For example:
    Dots in +x and +y represents drives when he went slower than his teammate but finished worse than teammate
    Dots in -x +y represents drives where he went faster on average but still finished lower, possibly showing bias
        (or inferiority)
    Uses driverconstructor.generate_teammate_diff_comparison_scatter.
    :param positions_source: Positions source
    :param constructor_results: Constructor results
    :param driver_id: Driver ID
    :return: Teammate diff. comparison scatter, explanation div
    """
    kwargs = dict(include_year_labels=False,
                  include_race_labels=True)
    return driverconstructor.generate_teammate_diff_comparison_scatter(positions_source, constructor_results, driver_id,
                                                                       **kwargs)


def generate_error_layout(year_id, driver_id, constructor_id):
    """
    Generates an error layout in the event that the given driver didn't compete for the given constructor in the given
    year.
    :param year_id: Year
    :param driver_id: Driver ID
    :param constructor_id: Constructor ID
    :return: Div layout
    """
    logging.info("Generating error layout")
    driver_name = get_driver_name(driver_id)
    constructor_name = get_constructor_name(constructor_id)
    year_races = races[races["year"] == year_id]
    year_results = results[results["raceId"].isin(year_races.index.values)]
    driver_results = year_results[year_results["driverId"] == driver_id]
    constructor_results = year_results[year_results["constructorId"] == constructor_id]

    # Generate the text
    text = f"Unfortunately, {driver_name} did not compete for {constructor_name} in {year_id}.<br>"

    text += f"{driver_name} competed for the following constructors this year:<br>"
    text += "<ul>"
    for cid in driver_results["constructorId"].unique():
        text += f"<li>{get_constructor_name(cid)}</li>"
    text += "</ul><br>"
    if constructor_results.shape[0] == 0:
        text += f"{constructor_name} did not compete this season.<br>"
    else:
        text += f"{constructor_name} had the following drivers this year:<br>"
        text += "<ul>"
        for did in constructor_results["driverId"].unique():
            text += f"<li>{get_driver_name(did)}</li>"
        text += "</ul><br>"
    text += f"The following constructors participated in the {year_id} season:<br>"
    text += "<ul>"
    for cid in year_results["constructorId"].unique():
        text += f"<li>{get_constructor_name(cid)}</li>"
    text += "</ul><br>"
    text += f"The following drivers participated in the {year_id} season:<br>"
    text += "<ul>"
    for did in year_results["driverId"].unique():
        text += f"<li>{get_driver_name(did)}</li>"
    text += "</ul><br>"
    layout = Div(text=text)
    return layout


def is_valid_input(year_id, driver_id, constructor_id):
    """
    Returns whether the given combo of year, driver, and constructor ID is valid.
    :param year_id: Year ID
    :param driver_id: Circuit ID
    :param constructor_id: Constructor ID
    :return: True if valid, False otherwise
    """
    year_races = races[races["year"] == year_id]
    year_rids = year_races.index.values
    year_results = results[results["raceId"].isin(year_rids)]
    yd_results = year_results[year_results["driverId"] == driver_id]

    return yd_results.shape[0] > 0

