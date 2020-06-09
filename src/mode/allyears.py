from bokeh.layouts import column
from bokeh.models import Div, Range1d, HoverTool, CrosshairTool, FixedTicker, DatetimeTickFormatter, Legend, LegendItem, \
    ColumnDataSource, LinearAxis, Label
import pandas as pd
from bokeh.plotting import figure
import numpy as np
from data_loading.data_loader import load_races, load_fastest_lap_data
from utils import DATETIME_TICK_KWARGS

races = load_races()
fastest_lap_data = load_fastest_lap_data()


def get_layout(**kwargs):
    years = races["year"].unique()
    years.sort()

    num_races_plot = generate_num_races_plot()

    speed_times_plot = generate_avg_speed_plot(years)

    header = Div(text=u"<h2><b>All Years \u2014 A Quick Summary of Formula 1</h2></b>")

    middle_spacer = Div()
    layout = column([
        header,
        num_races_plot, middle_spacer,
        speed_times_plot, middle_spacer,
    ], sizing_mode="stretch_width")

    return layout


def generate_num_races_plot():
    """
    Simple plot showing number of races vs year.
    :return: Plot layout
    """
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
        LegendItem(label="Circuit de Monaco", renderers=[monaco_speed_line]),
        LegendItem(label="Autodromo Nazionale di Monza", renderers=[monza_speed_line]),
        LegendItem(label="Silverstone Circuit", renderers=[silverstone_speed_line, label]),
        LegendItem(label="Circuit de Spa-Francorchamps", renderers=[spa_speed_line])
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


def generate_wdc_margin_plot(years):
    for year in years:
        # TODO make this method
        # Get final race ID
        # Get final WDC standings
        # Get points of 1st and 2nd place finishers
        # Calculate win margin as 1st pts / 2nd pts
        pass
    # TODO
    #  make a scatter plot
    #  x axis is year
    #  label each dot with the winner, or possibly color
    #  try connecting the dots, see if it makes any sense
    #  add crosshair
    #  add title, axes, etc
    #  add hover tooltip with winner name, constructor, win margin, year, 2nd and 3rd place
