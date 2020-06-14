import itertools
import logging
import math
from bokeh.layouts import column, row
from bokeh.models import Div, Range1d, NumeralTickFormatter, FixedTicker, HoverTool, CrosshairTool, Title, Label, \
    DataTable, ColumnDataSource, TableColumn
from bokeh.palettes import Turbo256
from bokeh.plotting import figure
import numpy as np
from bokeh.transform import cumsum
from data_loading.data_loader import load_drivers, load_results, load_races, load_wdc_final_positions, \
    load_fastest_lap_data
import pandas as pd
from utils import ColorDashGenerator, get_driver_name, int_to_ordinal, get_constructor_name, rounds_to_str, \
    get_status_classification, generate_spacer_item, generate_div_item, generate_plot_list_selector, PlotItem

drivers = load_drivers()
results = load_results()
races = load_races()
wdc_final_positions = load_wdc_final_positions()
fastest_lap_data = load_fastest_lap_data()

# TODO
#  1950 dnf rate: 49.375%
#  1965 dnf rate: 39.0%  √
#  1980 dnf rate: 44.6%
#  2010 dnf rate: 25.6%  √
#  2019 dnf rate: 13.5%  √


def get_layout(**kwargs):
    logging.info(f"Generating layout for mode ALLDRIVERS in alldrivers")

    description = u"Drivers Win Plot \u2014 plots the win percentage vs years of experience for every driver who has " \
                  u"won at least one race"
    all_drivers_win_plot = PlotItem(generate_all_drivers_win_plot, [], description)

    description = u"Start Position vs Finish Position Scatter Plot \u2014 each dot on this plot represents a " \
                  u"driver's career average start position vs finish position, and can show drivers who often made " \
                  u"up many places during races"
    spvfp_scatter = PlotItem(generate_career_spvfp_scatter, [], description)

    description = u"Start Position vs WDC Position Scatter Plot \u2014 each dot on this plot represents a driver's " \
                  u"career average start position vs World Drivers' Championship finish position, and is more " \
                  u"outlier-resistant than the above plot"
    sp_position_scatter = PlotItem(generate_sp_position_scatter, [], description)

    description = u"Average Lap Time Rank vs Finish Position Scatter Plot \u2014 each dot on this plot represents " \
                  u"a driver's career, and can show drivers who often out-drove their car"
    mltr_fp_scatter = PlotItem(generate_career_mltr_fp_scatter, [], description)

    description = u"Average Lap Time Rank vs WDC Finish Position Scatter Plot \u2014 each dot on this plot " \
                  u"represents a driver's career, and provides a different perspective on drivers who often " \
                  u"out-drove their car"
    mltr_position_scatter = PlotItem(generate_career_mltr_position_scatter, [], description)

    description = u"DNF Percent vs Podium Percent Scatter Plot \u2014 each dot on this plot represents a driver's " \
                  u"career and can show drivers who often did well despite an unreliable car, or vice-versa"
    dnf_podium_scatter = PlotItem(generate_dnf_podium_scatter, [], description)

    description = u"DNF Percent vs Win Percent Scatter Plot \u2014 each dot on this plot represents a driver's " \
                  u"career and can show drivers who often did well despite an unreliable car, or vice-versa " \
                  u"(a different perspective)"
    dnf_win_scatter = PlotItem(generate_dnf_win_scatter, [], description)

    description = u"DNF Percent vs Avg. Finish Position Scatter Plot \u2014 each dot on this plot represents a " \
                  u"driver's career and can show drivers who often did well despite an unreliable car, or vice-versa " \
                  u"(a third perspective)"
    dnf_fp_scatter = PlotItem(generate_dnf_fp_scatter, [], description)

    description = u"Winners Pie Chart \u2014 pie chart showing who won each race and World Drivers' Championship by " \
                  u"percentage"
    winners_pie = PlotItem(generate_winners_pie, [], description)

    description = u"All Drivers Table \u2014 table of every driver, including basic information such as who they " \
                  u"raced for, when they raced, and how well they did"
    drivers_table = PlotItem(generate_drivers_table, [], description)

    header = generate_div_item("<h2><b>All Drivers</b></h2>")

    middle_spacer = generate_spacer_item()
    group = generate_plot_list_selector([
        [header],
        [all_drivers_win_plot], [middle_spacer],
        [spvfp_scatter], [middle_spacer],
        [sp_position_scatter], [middle_spacer],
        [mltr_fp_scatter], [middle_spacer],
        [mltr_position_scatter], [middle_spacer],
        [dnf_podium_scatter], [middle_spacer],
        [dnf_win_scatter], [middle_spacer],
        [dnf_fp_scatter], [middle_spacer],
        [winners_pie],
        [drivers_table]
    ], header_addition="<br><i>Please note that </i><b>each</b><i> plot takes about 10 seconds to generate</i>")

    logging.info("Finished generating layout for mode ALLDRIVERS")

    return group


def generate_all_drivers_win_plot():
    """
    Generates a win plot including every driver.
    :return: All drivers win plot layout
    """
    logging.info("Generating all drivers win plot")

    winner_dids = results[results["position"] == 1]["driverId"].unique()

    win_plot = figure(
        title="Win Percentage of All Winning Drivers vs Years Experience",
        x_axis_label="Years of Experience (excluding breaks)",
        y_axis_label="Win Percentage",
        y_range=Range1d(0, 1, bounds=(0, 1))
    )
    win_plot.yaxis.formatter = NumeralTickFormatter(format="0.0%")
    win_plot.xaxis.ticker = FixedTicker(ticks=np.arange(0, 30, 1))

    palette = Turbo256
    n_drivers = len(winner_dids)
    colors = []
    di = 180 / n_drivers
    i = 40
    for _ in range(n_drivers):
        colors.append(palette[int(i)])
        i += di

    max_years_in = 0
    color_gen = ColorDashGenerator(colors=colors, driver_only_mode=True)
    for did in winner_dids:
        source = pd.DataFrame(columns=["years_in", "wins", "win_pct", "wins_str", "n_races"])
        years_in = 1
        driver_results = results[results["driverId"] == did]
        driver_races = races.loc[driver_results["raceId"]]
        driver_years = driver_races["year"].unique()
        driver_years.sort()
        wins = 0
        num_races = 0
        for year in driver_years:
            year_races = driver_races[driver_races["year"] == year]
            year_results = driver_results[driver_results["raceId"].isin(year_races.index.values)]
            num_races += year_results.shape[0]
            wins += year_results[year_results["position"] == 1].shape[0]
            win_pct = wins / num_races
            source = source.append({
                "driver_name": get_driver_name(did),
                "years_in": years_in,
                "wins": wins,
                "win_pct": win_pct,
                "n_races": num_races,
                "wins_str": str(wins) + " (" + str(100 * round(win_pct, 1)) + "%)"
            }, ignore_index=True)
            max_years_in = max(max_years_in, years_in)
            years_in += 1
        color, _ = color_gen.get_color_dash(did, did)
        win_plot.line(x="years_in", y="win_pct", source=source, color=color, alpha=0.6, line_width=2, muted_alpha=0.0)

    win_plot.x_range = Range1d(1, max_years_in, bounds=(1, max_years_in))

    win_plot.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Driver", "@driver_name"),
        ("Years into Career", "@years_in"),
        ("Wins", "@wins_str"),
        ("Number of Races", "@n_races")
    ]))

    win_plot.add_tools(CrosshairTool(dimensions="both", line_color="white", line_alpha=0.6))

    return win_plot


def generate_career_spvfp_scatter():
    """
    Generates average (across career) start position vs finish position scatter plot.
    :return: SP v FP scatter plot layout
    """
    logging.info("Generating start position vs finish position scatter")

    dids = drivers.index.values
    source = pd.DataFrame(columns=["driver_name", "avg_sp", "avg_fp", "num_races", "color", "size"])
    color_gen = itertools.cycle(Turbo256[25:220])
    for did in dids:
        driver_name = get_driver_name(did)
        driver_results = results[(results["driverId"] == did) & (results["grid"] > 0)]
        avg_sp = driver_results["grid"].mean()
        avg_fp = driver_results["position"].mean()
        num_races = driver_results.shape[0]
        size = math.pow(num_races, 0.3) + 2

        source = source.append({
            "driver_name": driver_name,
            "avg_sp": avg_sp,
            "avg_fp": avg_fp,
            "num_races": num_races,
            "color": color_gen.__next__(),
            "size": size
        }, ignore_index=True)

    spvfp_scatter = figure(title=u"Average Starting Position vs Finish Position \u2014 Saturday vs Sunday performance",
                           x_axis_label="Career Avg. Grid Position",
                           y_axis_label="Career Avg. Finishing Position (Official Classification)",
                           x_range=Range1d(0, 35, bounds=(0, 60)),
                           y_range=Range1d(0, 35, bounds=(0, 60)))
    spvfp_scatter.xaxis.ticker = FixedTicker(ticks=np.arange(5, 61, 5).tolist() + [1])
    spvfp_scatter.yaxis.ticker = FixedTicker(ticks=np.arange(5, 61, 5).tolist() + [1])
    spvfp_scatter.xaxis.major_label_overrides = {i: int_to_ordinal(i) for i in range(1, 60)}
    spvfp_scatter.yaxis.major_label_overrides = {i: int_to_ordinal(i) for i in range(1, 60)}

    subtitle = "Average is taken across the driver's whole career. Dot size is calculated based on the number of " \
               "races the driver entered. DNFs not considered in either calculation."
    spvfp_scatter.add_layout(Title(text=subtitle, text_font_style="italic"), "above")

    spvfp_scatter.scatter(x="avg_sp", y="avg_fp", source=source, color="color", size="size", alpha=0.7)
    spvfp_scatter.line(x=[-60, 60], y=[-60, 60], color="white", line_alpha=0.2)
    spvfp_scatter.line(x=[0, 60], y=[2.58240774, 29.96273108], color="white", line_alpha=0.6)  # Regression line

    label_kwargs = dict(render_mode="canvas",
                        text_color="white",
                        text_font_size="10pt",
                        border_line_color="white",
                        border_line_alpha=0.7)
    label1 = Label(x=32, y=16, text=" Regression Line ", **label_kwargs)
    label2 = Label(x=26, y=1, text=" Driver tends to make up many places ", **label_kwargs)
    label3 = Label(x=1, y=25, text=" Driver tends to lose many places ", **label_kwargs)
    spvfp_scatter.add_layout(label1)
    spvfp_scatter.add_layout(label2)
    spvfp_scatter.add_layout(label3)

    spvfp_scatter.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Driver", "@driver_name"),
        ("Avg. Starting Pos.", "@avg_sp"),
        ("Avg. Finish Pos.", "@avg_fp (DNFs not considered)"),
        ("Races Entered", "@num_races")
    ]))

    spvfp_scatter.add_tools(CrosshairTool(dimensions="both", line_color="white", line_alpha=0.6))

    return spvfp_scatter


def generate_sp_position_scatter():
    """
    Generates a scatter plot of average (across career) start position vs WDC position
    :return:
    """
    logging.info("Generating start position vs WDC finish position scatter")

    dids = drivers.index.values
    source = pd.DataFrame(columns=["driver_name", "avg_sp", "avg_wdc_position", "num_races", "color", "size"])
    color_gen = itertools.cycle(Turbo256[25:220])
    for did in dids:
        driver_name = get_driver_name(did)
        driver_results = results[(results["driverId"] == did) & (results["grid"] > 0)]
        avg_sp = driver_results["grid"].mean()
        avg_position = wdc_final_positions[wdc_final_positions["driverId"] == did]["position"].mean()

        num_races = driver_results.shape[0]
        size = math.pow(num_races, 0.3) + 2

        source = source.append({
            "driver_name": driver_name,
            "avg_sp": avg_sp,
            "avg_wdc_position": avg_position,
            "num_races": num_races,
            "color": color_gen.__next__(),
            "size": size
        }, ignore_index=True)

    sp_wdc_pos_scatter = figure(title=u"Average Starting Position vs WDC Finish Position \u2014 Saturday vs Sunday "
                                      u"performance, a less outlier-prone perspective",
                                x_axis_label="Career Avg. Grid Position",
                                y_axis_label="Career Avg. WDC Finish Position",
                                x_range=Range1d(0, 35, bounds=(0, 60)),
                                y_range=Range1d(0, 35, bounds=(0, 200)))
    sp_wdc_pos_scatter.xaxis.ticker = FixedTicker(ticks=np.arange(5, 61, 5).tolist() + [1])
    sp_wdc_pos_scatter.yaxis.ticker = FixedTicker(ticks=np.arange(5, 201, 5).tolist() + [1])
    sp_wdc_pos_scatter.xaxis.major_label_overrides = {i: int_to_ordinal(i) for i in range(1, 70)}
    sp_wdc_pos_scatter.yaxis.major_label_overrides = {i: int_to_ordinal(i) for i in range(1, 200)}

    subtitle = "Average is taken across the driver's whole career. Dot size is calculated based on the number of " \
               "races the driver entered."
    sp_wdc_pos_scatter.add_layout(Title(text=subtitle, text_font_style="italic"), "above")

    sp_wdc_pos_scatter.scatter(x="avg_sp", y="avg_wdc_position", source=source, color="color", size="size", alpha=0.7)
    sp_wdc_pos_scatter.line(x=[-60, 60], y=[-60, 60], color="white", line_alpha=0.2)

    # This is the correct regression line it just isn't very helpful on this plot
    # sp_wdc_pos_scatter.line(x=[0, 200], y=[7.65286499, 345.88496942], color="white", line_alpha=0.6)

    label_kwargs = dict(render_mode="canvas",
                        text_color="white",
                        text_font_size="10pt",
                        border_line_color="white",
                        border_line_alpha=0.7)
    # TODO are these really the right labels?
    label1 = Label(x=10, y=1, text=" Driver tends to make up many places ", **label_kwargs)
    label2 = Label(x=1, y=25, text=" Driver tends to lose many places ", **label_kwargs)
    sp_wdc_pos_scatter.add_layout(label1)
    sp_wdc_pos_scatter.add_layout(label2)

    sp_wdc_pos_scatter.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Driver", "@driver_name"),
        ("Avg. Starting Pos.", "@avg_sp"),
        ("Avg. WDC Pos.", "@avg_wdc_position"),
        ("Races Entered", "@num_races")
    ]))

    sp_wdc_pos_scatter.add_tools(CrosshairTool(dimensions="both", line_color="white", line_alpha=0.6))

    return sp_wdc_pos_scatter


def generate_career_mltr_fp_scatter():
    """
    Generates a scatter plot of mean lap time rank vs finish position (averaged across whole career).
    :return: Mean lap time rank vs finish position scatter layout
    """
    logging.info("Generating mean lap time rank vs finish position scatter")

    dids = drivers.index.values
    source = pd.DataFrame(columns=["driver_name", "avg_mltr", "avg_fp", "num_races", "color", "size"])
    color_gen = itertools.cycle(Turbo256[25:220])
    for did in dids:
        driver_name = get_driver_name(did)
        driver_results = results[(results["driverId"] == did) & (results["grid"] > 0)]
        avg_mltr = fastest_lap_data[fastest_lap_data["driver_id"] == did]["avg_lap_time_rank"].mean()
        avg_fp = driver_results["position"].mean()

        num_races = driver_results.shape[0]
        size = math.pow(num_races, 0.3) + 2

        source = source.append({
            "driver_name": driver_name,
            "avg_mltr": avg_mltr,
            "avg_fp": avg_fp,
            "num_races": num_races,
            "color": color_gen.__next__(),
            "size": size
        }, ignore_index=True)

    mltr_fp_scatter = figure(title=u"Average Lap Time Rank vs Finish Position \u2014 which drivers out-drove their "
                                   u"cars",
                             x_axis_label="Career Avg. Lap Time Rank",
                             y_axis_label="Career Avg. Finish Position",
                             x_range=Range1d(0, 35, bounds=(0, 60)),
                             y_range=Range1d(0, 35, bounds=(0, 200)))
    mltr_fp_scatter.xaxis.ticker = FixedTicker(ticks=np.arange(5, 61, 5).tolist() + [1])
    mltr_fp_scatter.yaxis.ticker = FixedTicker(ticks=np.arange(5, 201, 5).tolist() + [1])
    mltr_fp_scatter.xaxis.major_label_overrides = {i: int_to_ordinal(i) for i in range(1, 70)}
    mltr_fp_scatter.yaxis.major_label_overrides = {i: int_to_ordinal(i) for i in range(1, 200)}

    subtitle = "Average is taken across the driver's whole career. Dot size is calculated based on the number of " \
               "races the driver entered. DNFs are not included."
    mltr_fp_scatter.add_layout(Title(text=subtitle, text_font_style="italic"), "above")

    mltr_fp_scatter.scatter(x="avg_mltr", y="avg_fp", source=source, color="color", size="size", alpha=0.7)
    mltr_fp_scatter.line(x=[-60, 60], y=[-60, 60], color="white", line_alpha=0.6)

    label_kwargs = dict(render_mode="canvas",
                        text_color="white",
                        text_font_size="10pt",
                        border_line_color="white",
                        border_line_alpha=0.7)
    label1 = Label(x=10, y=1, text=" Driver tends to finish higher than expected ", **label_kwargs)
    label2 = Label(x=1, y=25, text=" Driver tends to finish lower than expected ", **label_kwargs)
    mltr_fp_scatter.add_layout(label1)
    mltr_fp_scatter.add_layout(label2)

    mltr_fp_scatter.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Driver", "@driver_name"),
        ("Avg. Lap Time Rank", "@avg_mltr"),
        ("Avg. Finish Position.", "@avg_fp"),
        ("Races Entered", "@num_races")
    ]))

    mltr_fp_scatter.add_tools(CrosshairTool(dimensions="both", line_color="white", line_alpha=0.6))

    return mltr_fp_scatter


def generate_career_mltr_position_scatter():
    """
    Generates a scatter plot of mean lap time rank vs WDC position (averaged across whole career).
    :return: Mean lap time rank vs position scatter layout
    """
    logging.info("Generating mean lap time rank vs WDC position scatter")

    dids = drivers.index.values
    source = pd.DataFrame(columns=["driver_name", "avg_mltr", "avg_position", "num_races", "color", "size"])
    color_gen = itertools.cycle(Turbo256[25:220])
    for did in dids:
        driver_name = get_driver_name(did)
        driver_results = results[(results["driverId"] == did) & (results["grid"] > 0)]
        avg_mltr = fastest_lap_data[fastest_lap_data["driver_id"] == did]["avg_lap_time_rank"].mean()
        avg_position = wdc_final_positions[wdc_final_positions["driverId"] == did]["position"].mean()

        num_races = driver_results.shape[0]
        size = math.pow(num_races, 0.3) + 2

        source = source.append({
            "driver_name": driver_name,
            "avg_mltr": avg_mltr,
            "avg_position": avg_position,
            "num_races": num_races,
            "color": color_gen.__next__(),
            "size": size
        }, ignore_index=True)

    mltr_position_scatter = figure(title=u"Average Lap Time Rank vs WDC Position \u2014 which drivers out-drove their "
                                   u"cars, a different perspective",
                                   x_axis_label="Career Avg. Lap Time Rank",
                                   y_axis_label="Career Avg. WDC Position",
                                   x_range=Range1d(0, 35, bounds=(0, 60)),
                                   y_range=Range1d(0, 35, bounds=(0, 200)))
    mltr_position_scatter.xaxis.ticker = FixedTicker(ticks=np.arange(5, 61, 5).tolist() + [1])
    mltr_position_scatter.yaxis.ticker = FixedTicker(ticks=np.arange(5, 201, 5).tolist() + [1])
    mltr_position_scatter.xaxis.major_label_overrides = {i: int_to_ordinal(i) for i in range(1, 70)}
    mltr_position_scatter.yaxis.major_label_overrides = {i: int_to_ordinal(i) for i in range(1, 200)}

    subtitle = "Average is taken across the driver's whole career. Dot size is calculated based on the number of " \
               "races the driver entered. DNFs are not included."
    mltr_position_scatter.add_layout(Title(text=subtitle, text_font_style="italic"), "above")

    mltr_position_scatter.scatter(x="avg_mltr", y="avg_position", source=source, color="color", size="size", alpha=0.7)
    mltr_position_scatter.line(x=[-60, 60], y=[-60, 60], color="white", line_alpha=0.6)

    label_kwargs = dict(render_mode="canvas",
                        text_color="white",
                        text_font_size="10pt",
                        border_line_color="white",
                        border_line_alpha=0.7)
    label1 = Label(x=10, y=1, text=" Driver tends to finish higher than expected ", **label_kwargs)
    label2 = Label(x=1, y=25, text=" Driver tends to finish lower than expected ", **label_kwargs)
    mltr_position_scatter.add_layout(label1)
    mltr_position_scatter.add_layout(label2)

    mltr_position_scatter.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Driver", "@driver_name"),
        ("Avg. Lap Time Rank", "@avg_mltr"),
        ("Avg. WDC Position.", "@avg_position"),
        ("Races Entered", "@num_races")
    ]))

    mltr_position_scatter.add_tools(CrosshairTool(dimensions="both", line_color="white", line_alpha=0.6))

    return mltr_position_scatter


def generate_dnf_podium_scatter():
    """
    Generates a scatter plot of DNF rate vs podium rate.
    :return: DNF vs podium scatter plot
    """
    logging.info("Generating DNF rate vs podium rate scatter")

    dids = drivers.index.values
    source = pd.DataFrame(columns=["driver_name",
                                   "podium_pct", "podium_pct_str",
                                   "dnf_pct", "dnf_pct_str",
                                   "num_races", "color", "size"])
    color_gen = itertools.cycle(Turbo256[25:220])
    for did in dids:
        driver_name = get_driver_name(did)
        driver_results = results[(results["driverId"] == did) & (results["grid"] > 0)]
        num_races = driver_results.shape[0]
        if num_races == 0:
            continue
        num_podiums = driver_results[driver_results["position"].isin([1, 2, 3])].shape[0]
        classifications = driver_results["statusId"].apply(get_status_classification)
        num_dnf = classifications[(classifications == "mechanical") | (classifications == "crash")].shape[0]

        podium_pct = num_podiums / num_races
        podium_pct_str = str(round(100 * podium_pct, 1)) + "%"
        podium_pct_str += " (" + str(num_podiums) + " / " + str(num_races) + " races entered)"

        dnf_pct = num_dnf / num_races
        dnf_pct_str = str(round(100 * dnf_pct, 1)) + "%"
        dnf_pct_str += " (" + str(num_dnf) + " / " + str(num_races) + " races entered)"

        size = math.pow(num_races, 0.3) + 2

        source = source.append({
            "driver_name": driver_name,
            "podium_pct": podium_pct,
            "podium_pct_str": podium_pct_str,
            "dnf_pct": dnf_pct,
            "dnf_pct_str": dnf_pct_str,
            "num_races": num_races,
            "color": color_gen.__next__(),
            "size": size
        }, ignore_index=True)

    dnf_podium_scatter = figure(title=u"DNF Percent vs. Podium Percent",
                                x_axis_label="DNF Percent (of races entered)",
                                y_axis_label="Podium Percent (of races entered)",
                                x_range=Range1d(0, 1, bounds=(0, 1)),
                                y_range=Range1d(0, 0.8, bounds=(0, 1)))
    dnf_podium_scatter.xaxis.formatter = NumeralTickFormatter(format="0.0%")
    dnf_podium_scatter.yaxis.formatter = NumeralTickFormatter(format="0.0%")

    subtitle = "Percentages are taken across the driver's whole career. Dot size is calculated based on the number " \
               "of races the driver entered."
    dnf_podium_scatter.add_layout(Title(text=subtitle, text_font_style="italic"), "above")

    dnf_podium_scatter.scatter(x="dnf_pct", y="podium_pct", source=source, color="color", size="size", alpha=0.7)

    dnf_podium_scatter.line(x=[.39] * 2, y=[-100, 100], color="white", line_width=2, alpha=0.4)
    dnf_podium_scatter.line(x=[.256] * 2, y=[-100, 100], color="white", line_width=2, alpha=0.4)
    dnf_podium_scatter.line(x=[.135] * 2, y=[-100, 100], color="white", line_width=2, alpha=0.4)

    label_kwargs = dict(render_mode="canvas",
                        text_color="white",
                        text_font_size="10pt",
                        border_line_color="white",
                        border_line_alpha=0.7)
    label1 = Label(x=0.1, y=0.7, text=" Often podiums, rare DNFs ", **label_kwargs)
    label2 = Label(x=0.7, y=0.2, text=" Rare podiums, often DNFs ", **label_kwargs)
    label3 = Label(x=0.6, y=0.6, text=" Often podiums, often DNFs ", **label_kwargs)
    label_kwargs["border_line_alpha"] = 0.0
    label4 = Label(x=0.39, y=.65, text=" 1950 Avg. DNF % ", **label_kwargs)
    label5 = Label(x=0.256, y=.65, text=" 2010 Avg. DNF % ", **label_kwargs)
    label6 = Label(x=0.135, y=.65, text=" 2019 Avg. DNF % ", **label_kwargs)
    dnf_podium_scatter.add_layout(label1)
    dnf_podium_scatter.add_layout(label2)
    dnf_podium_scatter.add_layout(label3)
    dnf_podium_scatter.add_layout(label4)
    dnf_podium_scatter.add_layout(label5)
    dnf_podium_scatter.add_layout(label6)

    dnf_podium_scatter.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Driver", "@driver_name"),
        ("DNF Percent", "@dnf_pct_str"),
        ("Podium Percent", "@podium_pct_str"),
        ("Races Entered", "@num_races")
    ]))

    dnf_podium_scatter.add_tools(CrosshairTool(dimensions="both", line_color="white", line_alpha=0.6))

    return dnf_podium_scatter


def generate_dnf_win_scatter():
    """
    Generates a scatter plot of DNF rate vs win rate.
    :return: DNF vs win scatter plot layout
    """
    logging.info("Generating DNF rate vs win rate scatter")

    dids = drivers.index.values
    source = pd.DataFrame(columns=["driver_name",
                                   "win_pct", "win_pct_str",
                                   "dnf_pct", "dnf_pct_str",
                                   "num_races", "color", "size"])
    color_gen = itertools.cycle(Turbo256[25:220])
    for did in dids:
        driver_name = get_driver_name(did)
        driver_results = results[(results["driverId"] == did) & (results["grid"] > 0)]
        num_races = driver_results.shape[0]
        if num_races == 0:
            continue
        num_wins = driver_results[driver_results["position"] == 1].shape[0]
        classifications = driver_results["statusId"].apply(get_status_classification)
        num_dnf = classifications[(classifications == "mechanical") | (classifications == "crash")].shape[0]

        win_pct = num_wins / num_races
        win_pct_str = str(round(100 * num_wins, 1)) + "%"
        win_pct_str += " (" + str(num_wins) + " / " + str(num_races) + " races entered)"

        dnf_pct = num_dnf / num_races
        dnf_pct_str = str(round(100 * dnf_pct, 1)) + "%"
        dnf_pct_str += " (" + str(num_dnf) + " / " + str(num_races) + " races entered)"

        size = math.pow(num_races, 0.3) + 2

        source = source.append({
            "driver_name": driver_name,
            "win_pct": win_pct,
            "win_pct_str": win_pct_str,
            "dnf_pct": dnf_pct,
            "dnf_pct_str": dnf_pct_str,
            "num_races": num_races,
            "color": color_gen.__next__(),
            "size": size
        }, ignore_index=True)

    dnf_win_scatter = figure(title=u"DNF Percent vs. Win Percent",
                             x_axis_label="DNF Percent (of races entered)",
                             y_axis_label="Win Percent (of races entered)",
                             x_range=Range1d(0, 1, bounds=(0, 1)),
                             y_range=Range1d(0, 0.6, bounds=(0, 1)))
    dnf_win_scatter.xaxis.formatter = NumeralTickFormatter(format="0.0%")
    dnf_win_scatter.yaxis.formatter = NumeralTickFormatter(format="0.0%")

    subtitle = "Percentages are taken across the driver's whole career. Dot size is calculated based on the number " \
               "of races the driver entered."
    dnf_win_scatter.add_layout(Title(text=subtitle, text_font_style="italic"), "above")

    dnf_win_scatter.scatter(x="dnf_pct", y="win_pct", source=source, color="color", size="size", alpha=0.7)

    dnf_win_scatter.line(x=[.39] * 2, y=[-100, 100], color="white", line_width=2, alpha=0.4)
    dnf_win_scatter.line(x=[.256] * 2, y=[-100, 100], color="white", line_width=2, alpha=0.4)
    dnf_win_scatter.line(x=[.135] * 2, y=[-100, 100], color="white", line_width=2, alpha=0.4)

    label_kwargs = dict(render_mode="canvas",
                        text_color="white",
                        text_font_size="10pt",
                        border_line_color="white",
                        border_line_alpha=0.7)
    label1 = Label(x=0.1, y=0.5, text=" Often wins, rare DNFs ", **label_kwargs)
    label2 = Label(x=0.7, y=0.2, text=" Rare wins, often DNFs ", **label_kwargs)
    label3 = Label(x=0.6, y=0.5, text=" Often wins, often DNFs ", **label_kwargs)
    label_kwargs["border_line_alpha"] = 0.0
    label4 = Label(x=0.39, y=.55, text=" 1950 Avg. DNF % ", **label_kwargs)
    label5 = Label(x=0.256, y=.55, text=" 2010 Avg. DNF % ", **label_kwargs)
    label6 = Label(x=0.135, y=.55, text=" 2019 Avg. DNF % ", **label_kwargs)
    dnf_win_scatter.add_layout(label1)
    dnf_win_scatter.add_layout(label2)
    dnf_win_scatter.add_layout(label3)
    dnf_win_scatter.add_layout(label4)
    dnf_win_scatter.add_layout(label5)
    dnf_win_scatter.add_layout(label6)

    dnf_win_scatter.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Driver", "@driver_name"),
        ("DNF Percent", "@dnf_pct_str"),
        ("Win Percent", "@win_pct_str"),
        ("Races Entered", "@num_races")
    ]))

    dnf_win_scatter.add_tools(CrosshairTool(dimensions="both", line_color="white", line_alpha=0.6))

    return dnf_win_scatter


def generate_dnf_fp_scatter():
    """
    Generates a scatter plot of DNF rate vs average finish position.
    :return: DNF vs win scatter plot layout
    """
    logging.info("Generating DNF rate vs average position scatter")

    dids = drivers.index.values
    source = pd.DataFrame(columns=["driver_name",
                                   "win_pct", "win_pct_str",
                                   "avg_fp",
                                   "num_races", "color", "size"])
    color_gen = itertools.cycle(Turbo256[25:220])
    for did in dids:
        driver_name = get_driver_name(did)
        driver_results = results[(results["driverId"] == did) & (results["grid"] > 0)]
        num_races = driver_results.shape[0]
        if num_races == 0:
            continue
        classifications = driver_results["statusId"].apply(get_status_classification)
        num_dnf = classifications[(classifications == "mechanical") | (classifications == "crash")].shape[0]

        avg_fp = driver_results["position"].mean()

        dnf_pct = num_dnf / num_races
        dnf_pct_str = str(round(100 * dnf_pct, 1)) + "%"
        dnf_pct_str += " (" + str(num_dnf) + " / " + str(num_races) + " races entered)"

        size = math.pow(num_races, 0.3) + 2

        source = source.append({
            "driver_name": driver_name,
            "avg_fp": avg_fp,
            "dnf_pct": dnf_pct,
            "dnf_pct_str": dnf_pct_str,
            "num_races": num_races,
            "color": color_gen.__next__(),
            "size": size
        }, ignore_index=True)

    dnf_fp_scatter = figure(title=u"DNF Percent vs. Average Finish Position",
                            x_axis_label="DNF Percent (of races entered)",
                            y_axis_label="Career Avg. Finish Position",
                            x_range=Range1d(0, 1, bounds=(0, 1)),
                            y_range=Range1d(0, 24, bounds=(0, 60)))
    dnf_fp_scatter.xaxis.formatter = NumeralTickFormatter(format="0.0%")
    dnf_fp_scatter.yaxis.ticker = FixedTicker(ticks=np.arange(5, 61, 5).tolist() + [1])
    dnf_fp_scatter.xaxis.major_label_overrides = {i: int_to_ordinal(i) for i in range(1, 60)}
    subtitle = "Percentages are taken across the driver's whole career. Dot size is calculated based on the number " \
               "of races the driver entered. DNFs are not considered for the average finish position."
    dnf_fp_scatter.add_layout(Title(text=subtitle, text_font_style="italic"), "above")

    dnf_fp_scatter.scatter(x="dnf_pct", y="avg_fp", source=source, color="color", size="size", alpha=0.7)

    dnf_fp_scatter.line(x=[.39] * 2, y=[-100, 100], color="white", line_width=2, alpha=0.4)
    dnf_fp_scatter.line(x=[.256] * 2, y=[-100, 100], color="white", line_width=2, alpha=0.4)
    dnf_fp_scatter.line(x=[.135] * 2, y=[-100, 100], color="white", line_width=2, alpha=0.4)

    label_kwargs = dict(render_mode="canvas",
                        text_color="white",
                        text_font_size="10pt",
                        border_line_color="white",
                        border_line_alpha=0.7)
    label1 = Label(x=0.1, y=1, text=" Often finishes well, rare DNFs ", **label_kwargs)
    label2 = Label(x=0.6, y=1, text=" Often finishes well, often DNFs ", **label_kwargs)
    label3 = Label(x=0.8, y=20, text=" Often finishes poorly, often DNFs ", **label_kwargs)
    label_kwargs["border_line_alpha"] = 0.0
    label4 = Label(x=0.39, y=22, text=" 1950 Avg. DNF % ", **label_kwargs)
    label5 = Label(x=0.256, y=22, text=" 2010 Avg. DNF % ", **label_kwargs)
    label6 = Label(x=0.135, y=22, text=" 2019 Avg. DNF % ", **label_kwargs)
    dnf_fp_scatter.add_layout(label1)
    dnf_fp_scatter.add_layout(label2)
    dnf_fp_scatter.add_layout(label3)
    dnf_fp_scatter.add_layout(label4)
    dnf_fp_scatter.add_layout(label5)
    dnf_fp_scatter.add_layout(label6)

    dnf_fp_scatter.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Driver", "@driver_name"),
        ("DNF Percent", "@dnf_pct_str"),
        ("Avg. Finish Position", "@avg_fp"),
        ("Races Entered", "@num_races")
    ]))

    dnf_fp_scatter.add_tools(CrosshairTool(dimensions="both", line_color="white", line_alpha=0.6))

    return dnf_fp_scatter


def generate_winners_pie():
    """
    Generates a pie chart of every winner of a race and WDC
    :return: Pie chart layout
    """
    logging.info("Generating winners pie plot")
    race_wins = results[results["position"] == 1]
    wins_source = race_wins.groupby("driverId").agg("count").rename(columns={"raceId": "num_wins"})["num_wins"]
    wins_source = pd.DataFrame(wins_source)
    wins_source["pct_wins"] = wins_source["num_wins"] / wins_source["num_wins"].sum()
    wins_source["pct_wins_str"] = wins_source["pct_wins"].apply(lambda x: str(100 * round(x, 1)) + "%")
    wins_source["angle"] = 2 * math.pi * wins_source["num_wins"] / wins_source["num_wins"].sum()
    wins_source["color"] = None
    wins_source["driver_name"] = None
    wins_source["constructor_name"] = None
    wins_source["alpha"] = None
    gen = ColorDashGenerator(dashes=[1, 0.5])
    for idx, source_row in wins_source.iterrows():
        did = idx
        cid = results[results["driverId"] == did]["constructorId"].mode()
        if cid.shape[0] > 0:
            cid = cid.values[0]
        else:
            cid = 0
        color, alpha = gen.get_color_dash(did, cid)
        wins_source.loc[did, "color"] = color
        wins_source.loc[did, "alpha"] = alpha
        wins_source.loc[did, "driver_name"] = get_driver_name(did)
        cids = results[results["driverId"] == did]["constructorId"].unique().tolist()
        wins_source.loc[did, "constructor_name"] = ", ".join(map(get_constructor_name, cids))

    race_pie_chart = figure(title=u"Race Winners",
                            toolbar_location=None,
                            tools="hover",
                            tooltips="Name: @driver_name<br>"
                                     "Race Wins: @num_wins (@pct_wins_str)<br>"
                                     "Constructor: @constructor_name",
                            x_range=(-0.5, 0.5), y_range=(-0.5, 0.5))

    race_pie_chart.wedge(x=0, y=0, radius=0.4, start_angle=cumsum("angle", include_zero=True),
                         end_angle=cumsum("angle"), line_color="white", color="color",
                         source=wins_source, fill_alpha="alpha")
    race_pie_chart.axis.axis_label = None
    race_pie_chart.axis.visible = False
    race_pie_chart.grid.grid_line_color = None

    # WDC wins pie chart
    wdc_wins = wdc_final_positions[wdc_final_positions["position"] == 1]
    wins_source = wdc_wins.groupby("driverId").agg("count").rename(columns={"year": "num_wins"})["num_wins"]
    wins_source = pd.DataFrame(wins_source)
    wins_source["pct_wins"] = wins_source["num_wins"] / wins_source["num_wins"].sum()
    wins_source["pct_wins_str"] = wins_source["pct_wins"].apply(lambda x: str(100 * round(x, 1)) + "%")
    wins_source["angle"] = 2 * math.pi * wins_source["num_wins"] / wins_source["num_wins"].sum()
    wins_source["color"] = None
    wins_source["driver_name"] = None
    wins_source["constructor_name"] = None
    wins_source["alpha"] = None
    gen = ColorDashGenerator(dashes=[1, 0.5])
    for idx, source_row in wins_source.iterrows():
        did = idx
        cid = results[results["driverId"] == did]["constructorId"].mode()
        if cid.shape[0] > 0:
            cid = cid.values[0]
        else:
            cid = 0
        color, alpha = gen.get_color_dash(did, cid)
        wins_source.loc[did, "color"] = color
        wins_source.loc[did, "alpha"] = alpha
        wins_source.loc[did, "driver_name"] = get_driver_name(did)
        cids = results[results["driverId"] == did]["constructorId"].unique().tolist()
        wins_source.loc[did, "constructor_name"] = ", ".join(map(get_constructor_name, cids))

    wdc_pie_chart = figure(title=u"WDC Winners",
                           toolbar_location=None,
                           tools="hover",
                           tooltips="Name: @driver_name<br>"
                                    "WDC Wins: @num_wins (@pct_wins_str)<br>"
                                    "Constructor: @constructor_name",
                           x_range=(-0.5, 0.5), y_range=(-0.5, 0.5))

    wdc_pie_chart.wedge(x=0, y=0, radius=0.4, start_angle=cumsum("angle", include_zero=True),
                        end_angle=cumsum("angle"), line_color="white", color="color",
                        source=wins_source, fill_alpha="alpha")
    wdc_pie_chart.axis.axis_label = None
    wdc_pie_chart.axis.visible = False
    wdc_pie_chart.grid.grid_line_color = None

    return row([race_pie_chart, wdc_pie_chart], sizing_mode="stretch_width")


def generate_drivers_table():
    """
    Table of all drivers showing:
    Name
    Number of races
    Years raced
    Number of WDC wins
    Number of race wins
    Number of podiums
    Constructors raced for
    :return: Drivers table layout
    """
    logging.info("Generating drivers table")

    dids = drivers.index.values
    source = pd.DataFrame(columns=["driver_name", "num_races", "years",
                                   "num_wdc", "num_wins", "num_podiums",
                                   "constructors"])

    for did in dids:
        driver_name = get_driver_name(did)
        driver_results = results[(results["driverId"] == did) & (results["grid"] > 0)]
        num_races = driver_results.shape[0]
        if num_races == 0:
            continue
        years = races.loc[driver_results["raceId"].values]["year"].unique()
        years.sort()
        years = rounds_to_str(years)
        num_wins = driver_results[driver_results["position"] == 1].shape[0]
        num_podiums = num_wins + driver_results[(driver_results["position"] == 2) |
                                                (driver_results["position"] == 3)].shape[0]
        wdc_wins = wdc_final_positions[(wdc_final_positions["driverId"] == did) &
                                       (wdc_final_positions["position"] == 1)]
        if wdc_wins.shape[0] == 0:
            num_wdc = str(wdc_wins.shape[0])
        else:
            wdc_wins_years = wdc_wins["year"].unique()
            wdc_wins_years.sort()
            num_wdc = str(wdc_wins.shape[0]) + " (" + rounds_to_str(wdc_wins_years) + ")"
        constructors = ", ".join(list(map(get_constructor_name, driver_results["constructorId"].unique())))

        source = source.append({
            "driver_name": driver_name,
            "num_races": num_races,
            "years": years,
            "num_wdc": num_wdc,
            "num_wins": num_wins,
            "num_podiums": num_podiums,
            "constructors": constructors
        }, ignore_index=True)
    source = source.sort_values(by=["num_wdc", "num_wins", "num_podiums", "num_races"], ascending=False)

    drivers_columns = [
        TableColumn(field="driver_name", title="Driver", width=60),
        TableColumn(field="num_races", title="Races", width=40),
        TableColumn(field="years", title="Years", width=85),
        TableColumn(field="num_wdc", title="WDC Wins", width=75),
        TableColumn(field="num_wins", title="Wins", width=40),
        TableColumn(field="num_podiums", title="Podiums", width=40),
        TableColumn(field="constructors", title="Constructors", width=130)
    ]

    drivers_table = DataTable(source=ColumnDataSource(data=source), columns=drivers_columns,
                              index_position=None, min_height=530)

    title = Div(text=f"<h2><b>All Drivers</b></h2>")

    return column([title, drivers_table], sizing_mode="stretch_width")
