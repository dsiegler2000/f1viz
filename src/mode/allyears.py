import logging
import math
from collections import defaultdict
import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import Div, Range1d, HoverTool, CrosshairTool, FixedTicker, Legend, LegendItem, NumeralTickFormatter, \
    ColumnDataSource, LabelSet, Label, TableColumn, DataTable
from bokeh.palettes import Category20_20, Set3_12
from bokeh.plotting import figure
from data_loading.data_loader import load_races, load_fastest_lap_data, load_driver_standings, load_results, \
    load_lap_times
from utils import get_driver_name, get_constructor_name, ColorDashGenerator

races = load_races()
fastest_lap_data = load_fastest_lap_data()
driver_standings = load_driver_standings()
results = load_results()
lap_times = load_lap_times()


def get_layout(**kwargs):
    years = races["year"].unique()
    years.sort()

    logging.info(f"Generating layout for mode ALLYEARS in allyears")

    num_races_plot = generate_num_races_plot()

    speed_times_plot = generate_avg_speed_plot(years)

    wdc_margin_plot, wdc_table, wdc_winners_dict = generate_wdc_margin_plot_table(years)

    wdc_bar_plot = generate_wdc_bar_plot(wdc_winners_dict)

    num_overtakes_plot = generate_num_overtakes_plot(years)

    win_plot = generate_top_drivers_win_plot()

    header = Div(text=u"<h2><b>All Years \u2014 A Quick Summary of Formula 1</h2></b>")

    middle_spacer = Div()
    layout = column([
        header,
        num_races_plot, middle_spacer,
        speed_times_plot, middle_spacer,
        wdc_margin_plot, middle_spacer,
        wdc_bar_plot, middle_spacer,
        num_overtakes_plot, middle_spacer,
        win_plot,
        wdc_table
    ], sizing_mode="stretch_width")

    logging.info("Finished generating layout for mode ALLYEARS")

    return layout


def generate_num_races_plot():
    """
    Simple plot showing number of races vs year.
    :return: Plot layout
    """
    logging.info("Generating num races plot")
    source = pd.DataFrame(races["year"].value_counts().sort_index()).rename(columns={"year": "num_races"})
    source.index.name = "year"
    min_year = source.index.min()
    max_year = source.index.max()
    num_races_plot = figure(
        title="Number of Races Each Season",
        x_axis_label="Year",
        y_axis_label="Number of Races",
        x_range=Range1d(min_year, max_year, bounds=(min_year, max_year)),
        y_range=Range1d(0, 25, bounds=(0, 25))
    )
    num_races_plot.xaxis.ticker = FixedTicker(ticks=np.arange(min_year, max_year + 1, 3))

    num_races_plot.line(x="year", y="num_races", source=source, line_width=2, line_color="white")

    num_races_plot.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Year", "@year"),
        ("Num. Races this Year", "@num_races")
    ]))

    num_races_plot.add_tools(CrosshairTool(dimensions="both", line_color="white", line_alpha=0.6))

    return num_races_plot


def generate_avg_speed_plot(years):
    """
    Plot of the average speed of a few classic tracks
    :return: Average speed plot layout
    """
    logging.info("Generating average speed plot")
    cids = [6,   # Monaco
            14,  # Monza
            9,   # Silverstone
            13]  # Spa
    # Circuit ID: list of tuples like ((start_year, end_year), circuit_len) with length in KM
    circuit_lengths = {
        6: [((1950, 1950), 3.180),
            ((1955, 1971), 3.145),
            ((1972, 1972), 3.145),
            ((1973, 1975), 3.278),
            ((1976, 1985), 3.312),
            ((1986, 1996), 3.328),
            ((1997, 1997), 3.366),
            ((1998, 1999), 3.367),
            ((2000, 2003), 3.370),
            ((2004, 2014), 3.340),
            ((2015, 2019), 3.337)],

        14: [((1950, 1954), 6.300),
             ((1955, 1956), 10.000),
             ((1957, 1959), 5.750),
             ((1960, 1961), 10.000),
             ((1962, 1971), 5.750),
             ((1972, 1973), 5.775),
             ((1974, 1975), 5.780),
             ((1976, 1993), 5.800),
             ((1994, 1994), 5.800),
             ((1995, 1999), 5.770),
             ((2000, 2019), 5.793)],

        9: [((1950, 1951), 4.649),
            ((1952, 1973), 4.710),
            ((1975, 1985), 4.718),
            ((1987, 1990), 4.778),
            ((1991, 1993), 5.226),
            ((1994, 1995), 5.057),
            ((1996, 1996), 5.072),
            ((1997, 1999), 5.140),
            ((2000, 2009), 5.141),
            ((2010, 2010), 5.901),
            ((2011, 2019), 5.891)],

        13: [((1950, 1956), 14.120),
             ((1958, 1968), 14.100),
             ((1970, 1970), 14.100),
             ((1983, 1983), 6.949),
             ((1985, 1991), 6.940),
             ((1994, 1994), 6.974),
             ((1992, 1995), 7.001),
             ((1996, 2001), 6.968),
             ((2002, 2002), 6.963),
             ((2004, 2004), 6.973),
             ((2005, 2005), 6.976),
             ((2007, 2019), 7.004)]
    }

    # TODO refactor this to have a "circuit_name" column instead of monaco_speed, monza_speed, etc
    source = pd.DataFrame(columns=["year",
                                   "monaco_speed_int", "monaco_len",
                                   "monza_speed_int", "monza_len",
                                   "silverstone_speed_int", "silverstone_len",
                                   "spa_time_millis", "spa_len"])
    for year in years:
        year_races = races[races["year"] == year]
        monaco_speed_int = np.nan
        monaco_len = np.nan

        monza_speed_int = np.nan
        monza_len = np.nan

        silverstone_speed_int = np.nan
        silverstone_len = np.nan

        spa_speed_int = np.nan
        spa_len = np.nan
        for cid in cids:
            if cid in year_races["circuitId"].values:
                rid = year_races[year_races["circuitId"] == cid].index.values[0]
                race_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"] == rid]
                if race_fastest_lap_data.shape[0] == 0:
                    continue
                avg_lap_time = race_fastest_lap_data["avg_lap_time_millis"].mean()
                # Determine the length
                circuit_configs = circuit_lengths[cid]
                length = 0
                for entry in circuit_configs:
                    year = int(round(year))
                    if entry[0][1] >= year >= entry[0][0]:
                        length = entry[1]
                avg_speed = length / (avg_lap_time / (1000 * 60 * 60))
                avg_speed = round(avg_speed, 1)
                if cid == 6:
                    monaco_speed_int = avg_speed
                    monaco_len = length
                elif cid == 14:
                    monza_speed_int = avg_speed
                    monza_len = length
                elif cid == 9:
                    silverstone_speed_int = avg_speed
                    silverstone_len = length
                else:
                    spa_speed_int = avg_speed
                    spa_len = length
        source = source.append({
            "year": year,
            "monaco_speed_int": monaco_speed_int,
            "monaco_len": monaco_len,
            "monza_speed_int": monza_speed_int,
            "monza_len": monza_len,
            "silverstone_speed_int": silverstone_speed_int,
            "silverstone_len": silverstone_len,
            "spa_speed_int": spa_speed_int,
            "spa_len": spa_len
        }, ignore_index=True)

    min_year = years.min()
    max_year = years.max()
    avg_speed_plot = figure(
        title="Average Speed and Lap Time of Select Circuits",
        x_axis_label="Year",
        y_axis_label="Avg. Speed (KPH)",
        x_range=Range1d(min_year, max_year, bounds=(min_year, max_year)),
    )
    avg_speed_plot.xaxis.ticker = FixedTicker(ticks=np.arange(min_year, max_year + 1, 3))

    kwargs = dict(
        x="year",
        source=source,
        line_width=2,
        muted_alpha=0.0
    )

    monaco_speed_line = avg_speed_plot.line(y="monaco_speed_int", color="white", **kwargs)
    monza_speed_line = avg_speed_plot.line(y="monza_speed_int", color="yellow", **kwargs)
    silverstone_speed_line = avg_speed_plot.line(y="silverstone_speed_int", color="orange", **kwargs)
    spa_speed_line = avg_speed_plot.line(y="spa_speed_int", color="hotpink", **kwargs)

    label = avg_speed_plot.text(x=[2010.5], y=[95], text=["Race Red Flagged"],
                                text_color="orange", text_font_size="10pt", muted_alpha=0.0)

    legend = [
        LegendItem(label="Monaco", renderers=[monaco_speed_line]),
        LegendItem(label="Monza", renderers=[monza_speed_line]),
        LegendItem(label="Silverstone", renderers=[silverstone_speed_line, label]),
        LegendItem(label="Spa-Francorchamps", renderers=[spa_speed_line])
    ]
    legend = Legend(items=legend, location="top_right", glyph_height=15, spacing=2, inactive_fill_color="gray")
    avg_speed_plot.add_layout(legend, "right")
    avg_speed_plot.legend.click_policy = "mute"
    avg_speed_plot.legend.label_text_font_size = "12pt"

    avg_speed_plot.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Year", "@year"),
        ("Monaco Avg. Speed", "@monaco_speed_int KPH"),
        ("Monaco Circuit Len.", "@monaco_len KM"),
        ("Monza Avg. Speed", "@monza_speed_int KPH"),
        ("Monza Circuit Len.", "@monza_len KM"),
        ("Silverstone Avg. Speed", "@silverstone_speed_int KPH"),
        ("Silverstone Circuit Len.", "@silverstone_len KM"),
        ("Spa Avg. Speed", "@spa_speed_int KPH"),
        ("Spa Circuit Len.", "@spa_len KM")
    ]))

    avg_speed_plot.add_tools(CrosshairTool(dimensions="both", line_color="white", line_alpha=0.6))

    return avg_speed_plot


def generate_wdc_margin_plot_table(years):
    """
    Generates a scatter plot of year vs WDC winning margin, where margin is calculated as
    (1st place points - 2nd place points) / 1st place points. Also generates a table of the results of every WDC.
    :param years: Years
    :return: WDC margin plot layout, WDC winner's table, winners dict
    """
    logging.info("Generating WDC margin plot")
    source = pd.DataFrame(columns=["year", "win_margin_pct", "win_margin_str",
                                   "first_str", "second_str", "third_str",
                                   "color", "first_short_str"])
    table_source = pd.DataFrame(columns=["year", "first_str", "second_str", "third_str"])
    winners_dict = defaultdict(int)

    color_gen = ColorDashGenerator(colors=Category20_20)
    for year in years:
        # Get final race ID
        year_races = races[races["year"] == year]
        year_results = results[results["raceId"].isin(year_races.index)]
        final_rid = year_races["round"].idxmax()

        # Get final WDC standings
        final_driver_standings = driver_standings[driver_standings["raceId"] == final_rid]
        if final_driver_standings.shape[0] == 0:
            continue

        first_row = final_driver_standings[final_driver_standings["position"] == 1].iloc[0]
        first_name = get_driver_name(first_row["driverId"])
        first_points = first_row["points"]
        first_points = int(first_points) if abs(first_points - int(first_points)) < 0.01 else first_points
        cid = year_results[year_results["driverId"] == first_row["driverId"]]["constructorId"].mode().values[0]
        first_constructor = get_constructor_name(cid)
        first_str_short = get_driver_name(first_row["driverId"], include_flag=False, just_last=True)
        winners_dict[first_name] += 1

        second_row = final_driver_standings[final_driver_standings["position"] == 2].iloc[0]
        second_name = get_driver_name(second_row["driverId"])
        second_points = second_row["points"]
        second_points = int(second_points) if abs(second_points - int(second_points)) < 0.01 else second_points
        cid = year_results[year_results["driverId"] == second_row["driverId"]]["constructorId"].mode().values[0]
        second_constructor = get_constructor_name(cid)

        third_row = final_driver_standings[final_driver_standings["position"] == 3].iloc[0]
        third_name = get_driver_name(third_row["driverId"])
        third_points = third_row["points"]
        third_points = int(third_points) if abs(third_points - int(third_points)) < 0.01 else third_points
        cid = year_results[year_results["driverId"] == third_row["driverId"]]["constructorId"].mode().values[0]
        third_constructor = get_constructor_name(cid)

        win_margin_absolute = first_points - second_points
        win_margin_pct = win_margin_absolute / first_points

        color, _ = color_gen.get_color_dash(first_row["driverId"], first_row["driverId"])

        source = source.append({
            "year": year,
            "win_margin_pct": win_margin_pct,
            "win_margin_str": str(win_margin_absolute) + " pts., " + str(round(100 * win_margin_pct, 3)) + "%",
            "first_str": first_name + " at " + first_constructor + " (" + str(first_points) + " pts.)",
            "second_str": second_name + " at " + second_constructor + " (" + str(second_points) + " pts.)",
            "third_str": third_name + " at " + third_constructor + " (" + str(third_points) + " pts.)",
            "color": color,
            "first_short_str": first_str_short
        }, ignore_index=True)

        table_source = table_source.append({
            "year": year,
            "first_str": first_name + " (" + first_constructor + ")",
            "second_str": second_name + " (" + second_constructor + ")",
            "third_str": third_name + " (" + third_constructor + ")"
        }, ignore_index=True)

    min_year = years.min()
    max_year = years.max()
    wdc_margin_plot = figure(
        title=u"WDC Win Margin \u2014 by what percent did the winner win by (winning margin / winner's points)",
        x_axis_label="Year",
        y_axis_label="Win Percent",
        x_range=Range1d(min_year, max_year + 0.5, bounds=(min_year, max_year + 0.5)),
        y_range=Range1d(0, 0.6, bounds=(0, 0.6))
    )
    wdc_margin_plot.yaxis.formatter = NumeralTickFormatter(format="0.0%")
    wdc_margin_plot.xaxis.ticker = FixedTicker(ticks=np.arange(min_year, max_year + 1, 3))

    wdc_margin_plot.scatter(x="year", y="win_margin_pct", source=source, color="color", size=6)

    # Labels
    labels = LabelSet(x="year", y="win_margin_pct", x_offset=0, y_offset=0, source=ColumnDataSource(source),
                      text="first_short_str", text_color="color", text_font_size="10pt", angle=0.6 * math.pi / 2)
    wdc_margin_plot.add_layout(labels)

    wdc_margin_plot.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Year", "@year"),
        ("Winning Margin", "@win_margin_str"),
        ("WDC Winner", "@first_str"),
        ("2nd Place", "@second_str"),
        ("3rd Place", "@third_str")
    ]))

    wdc_margin_plot.add_tools(CrosshairTool(dimensions="both", line_color="white", line_alpha=0.6))

    wdc_winners_columns = [
        TableColumn(field="year", title="Year", width=50),
        TableColumn(field="first_str", title="Winner", width=150),
        TableColumn(field="second_str", title="2nd Place", width=150),
        TableColumn(field="third_str", title="3rd Place", width=150),
    ]

    wdc_winners_table = DataTable(source=ColumnDataSource(data=table_source), columns=wdc_winners_columns,
                                  index_position=None, height=530)
    title = Div(text=f"<h2><b>World Drivers' Championship Results</b></h2>")

    return wdc_margin_plot, column([title, wdc_winners_table], sizing_mode="stretch_width"), winners_dict


def generate_wdc_bar_plot(wdc_winners_dict, top_n=10):
    """
    Simple graph of number the top WDC winners and how many wins they have
    :param wdc_winners_dict: WDC winners dict
    :param top_n: Top N drivers to consider
    :return: WDC bar plot layout
    """
    wdc_winners_dict = pd.DataFrame.from_dict(wdc_winners_dict, orient="index").rename(columns={0: "count"})
    wdc_winners_dict = wdc_winners_dict.sort_values(by="count", ascending=False)
    wdc_winners_dict.index.name = "name"
    wdc_winners_dict = wdc_winners_dict.iloc[:top_n]
    wdc_winners_dict["color"] = Set3_12[:top_n]

    wdc_bar_plot = figure(
        title=u"WDC Winners \u2014 Some of the Top WDC Winners",
        plot_height=400,
        x_range=wdc_winners_dict.index.values,
        y_range=Range1d(0, 8),
        toolbar_location=None,
        tools="",
        tooltips="@name: @count"
    )
    wdc_bar_plot.xaxis.major_label_orientation = 0.6 * math.pi / 2

    wdc_bar_plot.vbar(x="name", top="count", source=wdc_winners_dict, width=0.8, color="color")

    return wdc_bar_plot


def generate_num_overtakes_plot(years):
    """
    Generates a plot of mean number of overtakes and mean position change for every year (per race).
    :param years: Years
    :return: Plot layout
    """
    num_overtakes_dict = {
        1990: 30.9,
        1991: 30.9,
        1992: 25.4,
        1993: 24.5,
        1994: 18.1,
        1995: 17.5,
        1996: 11.6,
        1997: 15.6,
        1998: 12.9,
        1999: 16.3,
        2000: 16.4,
        2001: 13.5,
        2002: 13.8,
        2003: 18.9,
        2004: 16.2,
        2005: 10.9,
        2006: 16.2,
        2007: 15.9,
        2008: 14.8,
        2009: 13.2,
        2010: 23.8,
        2011: 43.2,
        2012: 43.5,
        2013: 40.0,
        2014: 33.5,
        2015: 26.8,
        2016: 41.2
    }
    finished_results = results[~results["position"].isna()]
    source = pd.DataFrame(columns=["year", "avg_position_change", "avg_num_overtakes", "avg_num_overtakes_str"])
    for year in years:
        year_races = races[races["year"] == year]
        year_results = finished_results[finished_results["raceId"].isin(year_races.index)]
        avg_position_change = (year_results["grid"] - year_results["position"]).sum() / year_races.shape[0]
        avg_num_overtakes = np.nan
        if year in num_overtakes_dict.keys():
            avg_num_overtakes = num_overtakes_dict[year]
        source = source.append({
            "year": year,
            "avg_position_change": str(round(avg_position_change, 1)),
            "avg_num_overtakes": avg_num_overtakes,
            "avg_num_overtakes_str": str(round(avg_num_overtakes, 1)) if not np.isnan(avg_num_overtakes) else ""
        }, ignore_index=True)

    min_year = years.min()
    max_year = years.max()
    num_overtakes_plot = figure(
        title=u"Average Number of Overtakes and Position Changes per Race vs Season \u2014 See below for specifics",
        x_axis_label="Year",
        y_axis_label="Number",
        x_range=Range1d(min_year, max_year, bounds=(min_year, max_year)),
        y_range=Range1d(0, 80, bounds=(0, 80))
    )
    num_overtakes_plot.xaxis.ticker = FixedTicker(ticks=np.arange(min_year, max_year + 1, 3))

    kwargs = dict(
        x="year",
        source=source,
        line_width=2,
        muted_alpha=0.0
    )
    num_overtakes_line = num_overtakes_plot.line(y="avg_num_overtakes", color="white", **kwargs)
    num_pos_change_line = num_overtakes_plot.line(y="avg_position_change", color="orange", **kwargs)

    label = Label(x=2010.5, y=45, text="DRS Introduced", text_font_size="10pt",
                  angle=0.8 * math.pi / 2, text_color="white")
    num_overtakes_plot.add_layout(label)

    legend = [
        LegendItem(label="Avg. Overtakes / Race", renderers=[num_overtakes_line]),
        LegendItem(label="Avg. Position Changes / Race", renderers=[num_pos_change_line])
    ]
    legend = Legend(items=legend, location="top_right", glyph_height=15, spacing=2, inactive_fill_color="gray")
    num_overtakes_plot.add_layout(legend, "right")
    num_overtakes_plot.legend.click_policy = "mute"
    num_overtakes_plot.legend.label_text_font_size = "12pt"

    explanation_div = Div(text="""An overtake, or more precisely a "overtaking move" is classified as an overtake that 
    takes place during a flying lap (i.e. not the first lap) and is maintained all the way to the finish. Overtakes due 
    to DNFs, lapping, unlapping, and pit stops are not counted,
     <br>Average position change is simply computed as the average of all starting positions minus finish positions, not 
     including drivers who DNF'd.""")

    num_overtakes_plot.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Year", "@year"),
        ("Avg. Pos. Change per Race", "@avg_position_change"),
        ("Avg. Num. Overtakes per Race", "@avg_num_overtakes_str")
    ]))

    num_overtakes_plot.add_tools(CrosshairTool(dimensions="both", line_color="white", line_alpha=0.6))

    return column([num_overtakes_plot, explanation_div], sizing_mode="stretch_width")


def generate_top_drivers_win_plot(dids=None):
    logging.info("Generating top drivers win plot")
    if dids is None:
        dids = [
            30,   # Schumacher
            1,    # Hamilton
            579,  # Fangio
            102,  # Senna
            373   # Clark
        ]

    win_plot = figure(
        title="Win Percentage of Top Drivers vs Years Experience",
        x_axis_label="Years of Experience (excluding breaks)",
        y_axis_label="Win Percentage",
        y_range=Range1d(0, 0.5, bounds=(0, 0.5))
    )
    win_plot.yaxis.formatter = NumeralTickFormatter(format="0.0%")
    win_plot.xaxis.ticker = FixedTicker(ticks=np.arange(0, 30, 1))

    color_gen = ColorDashGenerator(driver_only_mode=True)
    legend = []
    max_years_in = 0
    for did in dids:
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
        line = win_plot.line(x="years_in", y="win_pct", source=source, color=color, line_width=2, muted_alpha=0.0)
        legend.append(LegendItem(label=get_driver_name(did), renderers=[line]))

    legend = Legend(items=legend, location="top_right", glyph_height=15, spacing=2, inactive_fill_color="gray")
    win_plot.add_layout(legend, "right")
    win_plot.legend.click_policy = "mute"
    win_plot.legend.label_text_font_size = "12pt"

    win_plot.x_range = Range1d(1, max_years_in, bounds=(1, max_years_in))

    win_plot.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Driver", "@driver_name"),
        ("Years into Career", "@years_in"),
        ("Wins", "@wins_str"),
        ("Number of Races", "@n_races")
    ]))

    win_plot.add_tools(CrosshairTool(dimensions="both", line_color="white", line_alpha=0.6))

    return win_plot


def generate_wcc_table():
    # todo this, see wdc table
    pass


