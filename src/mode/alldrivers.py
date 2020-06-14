import itertools
import logging
import math
from bokeh.layouts import column
from bokeh.models import Div, Range1d, NumeralTickFormatter, FixedTicker, HoverTool, CrosshairTool, Title, Label
from bokeh.palettes import Turbo256
from bokeh.plotting import figure
import numpy as np
from data_loading.data_loader import load_drivers, load_results, load_races, load_wdc_final_positions
import pandas as pd
from utils import ColorDashGenerator, get_driver_name, int_to_ordinal

# TODO potentially have generate_top_drivers_win_plot (from allyears) but include all drivers

drivers = load_drivers()
results = load_results()
races = load_races()
wdc_final_positions = load_wdc_final_positions()


def get_layout(**kwargs):
    logging.info(f"Generating layout for mode ALLDRIVERS in alldrivers")

    all_drivers_win_plot = Div(text="<h1><b><i>should I include driver win plot?</h2></b></i>")  # generate_all_drivers_win_plot()

    # todo add note that all plots take a while to generate

    spvfp_scatter = generate_career_spvfp_scatter()

    sp_position_scatter = generate_sp_position_scatter()

    header = Div(text="<h2><b>All Drivers</b></h2>")

    middle_spacer = Div()
    layout = column([
        header,
        all_drivers_win_plot, middle_spacer,
        spvfp_scatter, middle_spacer,
        sp_position_scatter, middle_spacer
    ], sizing_mode="stretch_width")

    logging.info("Finished generating layout for mode ALLDRIVERS")

    return layout


def generate_all_drivers_win_plot():
    """
    Generates a win plot including every driver.
    :return: All drivers win plot layout
    """
    # TODO should I include this?
    # TODO potentially try to implement on mute-on-click type thing
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
    n_circuits = len(winner_dids)
    colors = []
    di = 180 / n_circuits
    i = 20
    for _ in range(n_circuits):
        colors.append(palette[int(i)])
        i += di

    max_years_in = 0
    color_gen = ColorDashGenerator(colors=colors, driver_only_mode=True)
    n = len(winner_dids)
    i = 0
    for did in winner_dids:
        i += 1
        if i % 100 == 0:
            print(f"{i} / {n}")
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
    sp_wdc_pos_scatter.line(x=[-60, 60], y=[-60, 60], color="white", line_alpha=0.1)

    # This is the correct regression line it just isn't very helpyful on this plot
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
    # TODO do me next, mean lap time rank vs average finish position scatter (average across career)
    #  try with both "position" and "positionOrder" (see what the driver.generate_mltr_fp_scatter does)
    #  try with MLTR vs WDC average finish position
    #  try adding trend line
    pass


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
    :return:
    """
    pass
