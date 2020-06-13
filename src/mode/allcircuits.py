import logging
import math
from bokeh.layouts import column, row
from bokeh.models import Range1d, Div, NumeralTickFormatter, Title, LabelSet, ColumnDataSource, DatetimeTickFormatter, \
    HoverTool, FixedTicker, CrosshairTool, Label, PrintfTickFormatter, TableColumn, DataTable
from bokeh.palettes import Turbo256
from bokeh.plotting import figure
import pandas as pd
import numpy as np
from data_loading.data_loader import load_circuits, load_races, load_results, load_fastest_lap_data, load_pit_stops, \
    load_driver_standings, load_wdc_final_positions
from utils import get_circuit_name, rounds_to_str, get_status_classification, millis_to_str, DATETIME_TICK_KWARGS, \
    int_to_ordinal, get_driver_name, get_constructor_name

circuits = load_circuits()
circuits = circuits.drop(72)  # Drop Port Imperial Street Circuit as it was never raced on
circuits = circuits.drop(74)  # Drop Hanoi Street Circuit for now
races = load_races()
results = load_results()
fastest_lap_data = load_fastest_lap_data()
pit_stop_data = load_pit_stops()
driver_standings = load_driver_standings()
wdc_final_positions = load_wdc_final_positions()


def get_layout(**kwargs):
    years = races["year"].unique()
    years.sort()

    num_races_bar_chart = generate_num_races_bar()

    dnf_bar_chart = generate_dnf_bar()

    overtakes_bar_chart = generate_num_overtakes_bar()

    mspmfp_bar_chart = generate_mspmfp_bar()

    countries_bar_chart = generate_countries_bar()

    rating_bar_chart = generate_rating_bar()

    avg_lap_time_bar_chart = generate_avg_lap_time_bar()

    pit_stop_bar_chart = generate_pit_stop_bar()

    sc_laps_bar_chart = generate_sc_laps_bar()

    weather_bar_chart = generate_weather_bar()

    upset_scatter = generate_upset_scatter()

    spvfp_scatter = generate_spvfp_scatter()

    circuits_table = generate_circuits_table()

    header = Div(text=u"<h2><b>All Circuits \u2014 Some Stats on All Circuits that have held F1 Races")

    middle_spacer = Div()
    layout = column([
        header,
        num_races_bar_chart, middle_spacer,
        dnf_bar_chart, middle_spacer,
        overtakes_bar_chart, middle_spacer,
        mspmfp_bar_chart, middle_spacer,
        rating_bar_chart, middle_spacer,
        avg_lap_time_bar_chart, middle_spacer,
        pit_stop_bar_chart, middle_spacer,
        sc_laps_bar_chart, middle_spacer,
        weather_bar_chart, middle_spacer,
        countries_bar_chart, middle_spacer,
        row([upset_scatter, spvfp_scatter], sizing_mode="stretch_width"), middle_spacer,
        circuits_table
    ], sizing_mode="stretch_width")

    return layout


def generate_num_races_bar():
    """
    Generates a bar chart showing the number of races each circuit has held.
    :return: Number of races bar chart
    """
    logging.info("Generating num races bar plot")

    source = pd.DataFrame(races["circuitId"].value_counts()).rename(columns={"circuitId": "count"})
    source["circuit_name"] = source.index.map(get_circuit_name_custom)

    def get_years(cid):
        years = races[races["circuitId"] == cid]["year"].unique()
        years.sort()
        return rounds_to_str(years)
    source["years"] = source.index.map(get_years)

    num_races_bar = figure(
        title="Number of Races",
        x_range=source["circuit_name"].unique(),
        y_range=Range1d(0, 75, bounds=(0, 75)),
        plot_height=600,
        toolbar_location=None,
        tools="",
        tooltips="Circuit Name: @circuit_name<br>"
                 "Races Hosted: @count<br>"
                 "Years Held: @years"
    )
    num_races_bar.xaxis.major_label_orientation = math.pi / 2

    palette = Turbo256
    n_circuits = source.shape[0]
    colors = []
    di = 230 / n_circuits
    i = 20
    for _ in range(n_circuits):
        colors.append(palette[int(i)])
        i += di
    source["color"] = colors

    num_races_bar.vbar(x="circuit_name", top="count", source=source, width=0.8, color="color")

    return num_races_bar


def generate_countries_bar():
    """
    Generates a bar plot showing the number of circuits in every country
    :return: Countries bar plot layout
    """
    logging.info("Generating countries bar plot")

    source = pd.DataFrame(circuits["country"].value_counts()).rename(columns={"country": "count"})
    source.index.name = "country_name"

    def get_circuits(country):
        return ", ".join(circuits[circuits["country"] == country]["name"].unique().tolist())

    source["circuits"] = source.index.map(get_circuits)

    countries_bar = figure(
        title="Number of Circuits in Every Country",
        x_range=source.index.values,
        y_range=Range1d(0, 13, bounds=(0, 13)),
        plot_height=350,
        toolbar_location=None,
        tools="",
        tooltips="Country: @country_name<br>"
                 "Number of Circuits: @count<br>"
                 "Circuits in this country: @circuits"
    )
    countries_bar.xaxis.major_label_orientation = math.pi / 2

    palette = Turbo256
    n_circuits = source.shape[0]
    colors = []
    di = 230 / n_circuits
    i = 20
    for _ in range(n_circuits):
        colors.append(palette[int(i)])
        i += di
    source["color"] = colors

    countries_bar.vbar(x="country_name", top="count", source=source, width=0.8, color="color")

    return countries_bar


def generate_dnf_bar():
    """
    Generates a bar plot showing the DNF percent of every circuit
    :return: Bar plot layout
    """
    logging.info("Generating DNF percent bar plot")

    def get_dnf_pct(cid):
        circuit_races = races[races["circuitId"] == cid]
        circuit_results = results[results["raceId"].isin(circuit_races.index.values)]
        classifications = circuit_results["statusId"].apply(get_status_classification)
        dnfs = classifications[(classifications == "crashed") | (classifications == "mechanical")].shape[0]
        total = dnfs + classifications[classifications == "finished"].shape[0]
        return dnfs / total if total > 0 else 0
    source = pd.DataFrame(circuits.index.values, columns=["circuit_id"])
    source["dnf_pct"] = source["circuit_id"].apply(get_dnf_pct)
    source["dnf_pct_str"] = source["dnf_pct"].apply(lambda p: str(round(100 * p, 1)) + "%")
    source["circuit_name"] = source["circuit_id"].apply(get_circuit_name_custom)
    source = source.sort_values(by="dnf_pct", ascending=False)

    dnf_bar = figure(
        title="DNF Percent at Every Circuit (calculated as number of DNFs / number of starts)",
        y_axis_label="DNF Percent",
        x_range=source["circuit_name"].unique(),
        y_range=Range1d(0, 0.75, bounds=(0, 0.75)),
        plot_height=600,
        toolbar_location=None,
        tools="",
        tooltips="Circuit: @circuit_name<br>"
                 "DNF Percent: @dnf_pct_str"
    )
    dnf_bar.xaxis.major_label_orientation = math.pi / 2
    dnf_bar.yaxis.formatter = NumeralTickFormatter(format="0%")

    palette = Turbo256
    n_circuits = source.shape[0]
    colors = []
    di = 230 / n_circuits
    i = 20
    for _ in range(n_circuits):
        colors.append(palette[int(i)])
        i += di
    source["color"] = colors

    dnf_bar.vbar(x="circuit_name", top="dnf_pct", source=source, width=0.8, color="color")

    return dnf_bar


def generate_mspmfp_bar():
    """
    Generates a bar plot of mean start position minus finish position for each circuit, correcting for DNFs (though
    this DNF correction isn't working properly right now)
    :return: MSPMFP bar plot layout
    """
    logging.info("Generating mean start position minus finish position bar plot")

    source = pd.DataFrame()

    def get_mspmfp(cid):
        circuit_races = races[races["circuitId"] == cid]
        circuit_results = results[results["raceId"].isin(circuit_races.index.values)].copy()
        classifications = circuit_results["statusId"].apply(get_status_classification)
        circuit_results["classification"] = classifications
        dnf_results = circuit_results[(circuit_results["classification"] == "crashed") |
                                      (circuit_results["classification"] == "mechanical")]
        return (circuit_results["grid"] - circuit_results["position"]).mean() - \
               (dnf_results["positionOrder"] - dnf_results["grid"]).mean()

    # TODO this calculation still isn't quite right, both "mspmfp" and mspmfp * num_races doesn't sum up to 0
    source["mspmfp"] = circuits.index.map(get_mspmfp)
    source["circuit_name"] = circuits.index.map(get_circuit_name_custom)
    source["mspmfp_str"] = source["mspmfp"].apply(lambda x: str(abs(round(x, 1))) + " positions " +
                                                  ("higher" if x > 0 else "lower")) + " he/she they started"
    source["num_races"] = circuits.index.map(lambda cid: races[races["circuitId"] == cid].shape[0])
    source = source.sort_values(by="mspmfp", ascending=False)

    mspmfp_bar = figure(
        title=u"Average Starting Position Minus Finish Position \u2014 How many positions do drivers gain or lose on "
              u"average at this circuit",
        x_range=source["circuit_name"].unique(),
        y_axis_label="",
        y_range=Range1d(-10, 6, bounds=(-10, 6)),
        plot_height=600,
        toolbar_location=None,
        tools="",
        tooltips="Circuit: @circuit_name<br>"
                 "Average driver finishes @mspmfp_str<br>"
                 "Number of Races at this circuit: @num_races"
    )
    subtitle = f"Positions gained from another driver who has DNF'd are not counted"
    mspmfp_bar.add_layout(Title(text=subtitle, text_font_style="italic"), "above")
    mspmfp_bar.xaxis.major_label_orientation = math.pi / 2
    label_source = ColumnDataSource(dict(x=["Long Beach 🇺🇸"] * 2, y=[4, -5], text=[
        "On average, drivers finished higher than they started",
        "On average, drivers finished lower than they started"]))
    labels = LabelSet(x="x", y="y", text="text", source=label_source, text_color="white", text_font_size="12pt",
                      level="glyph", render_mode="canvas")
    mspmfp_bar.add_layout(labels)

    palette = Turbo256
    n_circuits = source.shape[0]
    colors = []
    di = 230 / n_circuits
    i = 20
    for _ in range(n_circuits):
        colors.append(palette[int(i)])
        i += di
    source["color"] = colors

    mspmfp_bar.vbar(x="circuit_name", top="mspmfp", source=source, width=0.8, color="color")

    return mspmfp_bar


def generate_rating_bar():
    """
    Generates a bar plot showing average rating of circuits that we have rating data on.
    :return: Ratings bar plot layout
    """
    logging.info("Generating rating bar plot")

    source = pd.DataFrame(races.groupby("circuitId")["rating"].mean())
    source = source[source["rating"].notna()]
    source["circuit_name"] = source.index.map(get_circuit_name_custom)
    source["rating_str"] = source["rating"].apply(lambda r: str(round(r, 1)) + " / 10")
    source = source.sort_values(by="rating", ascending=False)

    rating_bar = figure(
        title=u"Average Rating of Some Circuit Out of 10 \u2014 from racefans.net, only includes races from 2008-2017",
        y_axis_label="Rating",
        x_range=source["circuit_name"].unique(),
        y_range=Range1d(0, 10, bounds=(0, 10)),
        plot_height=600,
        toolbar_location=None,
        tools="",
        tooltips="Circuit: @circuit_name<br>"
                 "Average Rating: @rating_str"
    )
    rating_bar.xaxis.major_label_orientation = math.pi / 2

    palette = Turbo256
    n_circuits = source.shape[0]
    colors = []
    di = 230 / n_circuits
    i = 20
    for _ in range(n_circuits):
        colors.append(palette[int(i)])
        i += di
    source["color"] = colors

    rating_bar.vbar(x="circuit_name", top="rating", source=source, width=0.8, color="color")

    return rating_bar


def generate_avg_lap_time_bar():
    """
    Generates a bar plot showing average lap times.
    :return: Average lap time bar plot layout
    """
    logging.info("Generating average lap time bar plot")

    def get_avg_lap_time_millis(cid):
        circuit_races = races[races["circuitId"] == cid]
        circuit_fl_data = fastest_lap_data[fastest_lap_data["raceId"].isin(circuit_races.index.values)]
        return circuit_fl_data["avg_lap_time_millis"].mean()
    source = pd.DataFrame(circuits.index.values, columns=["circuit_id"])
    source["avg_lap_time_millis"] = source["circuit_id"].apply(get_avg_lap_time_millis)
    source["circuit_name"] = source["circuit_id"].apply(get_circuit_name_custom)
    source["avg_lap_time_str"] = source["avg_lap_time_millis"].apply(millis_to_str)
    source = source.sort_values(by="avg_lap_time_millis", ascending=False)

    avg_lap_time_bar = figure(
        title=u"Average Lap Time \u2014 All GPs at this circuit are included, not just recent ones",
        y_axis_label="Lap Time",
        x_range=source["circuit_name"].unique(),
        y_range=Range1d(1000, 400000, bounds=(1000, 400000)),
        plot_height=600,
        toolbar_location=None,
        tools="",
        tooltips="Circuit: @circuit_name<br>"
                 "Average Lap Time: @avg_lap_time_str"
    )
    avg_lap_time_bar.xaxis.major_label_orientation = math.pi / 2
    avg_lap_time_bar.yaxis.formatter = DatetimeTickFormatter(**DATETIME_TICK_KWARGS)

    palette = Turbo256
    n_circuits = source.shape[0]
    colors = []
    di = 230 / n_circuits
    i = 20
    for _ in range(n_circuits):
        colors.append(palette[int(i)])
        i += di
    source["color"] = colors

    avg_lap_time_bar.vbar(x="circuit_name", top="avg_lap_time_millis", source=source, width=0.8, color="color")

    return avg_lap_time_bar


def generate_pit_stop_bar():
    """
    Generates a bar plot showing the average number of pit stops per race.
    :return: Pit stop bar plot layout
    """
    logging.info("Generating pit stop bar plot")

    def get_avg_num_pit_stops(cid):
        circuit_races = races[races["circuitId"] == cid]
        circuit_ps_data = pit_stop_data[pit_stop_data["raceId"].isin(circuit_races.index.values)]
        if circuit_ps_data.shape[0] == 0:
            return np.nan
        else:
            return circuit_ps_data.shape[0] / circuit_ps_data["raceId"].unique().shape[0]

    source = pd.DataFrame(circuits.index.values, columns=["circuit_id"])
    source["avg_num_pit_stops"] = source["circuit_id"].apply(get_avg_num_pit_stops)
    source = source[(source["avg_num_pit_stops"].notna()) & (source["avg_num_pit_stops"] > 0)]
    source["circuit_name"] = source["circuit_id"].apply(get_circuit_name_custom)
    source["avg_num_pit_stops_str"] = source["avg_num_pit_stops"].apply(lambda x: str(round(x, 1)))
    source = source.sort_values(by="avg_num_pit_stops", ascending=False)

    pit_stops_bar = figure(
        title=u"Average Number of Pit Stops per Race (2012 onward)",
        y_axis_label="Avg. Num. Pit Stops",
        x_range=source["circuit_name"].unique(),
        y_range=Range1d(0, 90, bounds=(0, 90)),
        plot_height=600,
        toolbar_location=None,
        tools="",
        tooltips="Circuit: @circuit_name<br>"
                 "Average Number of Pit Stops: @avg_num_pit_stops_str"
    )
    pit_stops_bar.xaxis.major_label_orientation = math.pi / 2

    palette = Turbo256
    n_circuits = source.shape[0]
    colors = []
    di = 230 / n_circuits
    i = 20
    for _ in range(n_circuits):
        colors.append(palette[int(i)])
        i += di
    source["color"] = colors

    pit_stops_bar.vbar(x="circuit_name", top="avg_num_pit_stops", source=source, width=0.8, color="color")

    return pit_stops_bar


def generate_upset_scatter():
    """
    Generates a scatter plot showing how prone circuits are to upsets.
    :return: Upset plot layout
    """
    logging.info("Generating upset scatter plot")

    def get_winner_wdc_position(cid):
        # Gets the mean WDC position for the winner at this circuit
        circuit_races = races[races["circuitId"] == cid]
        circuit_results = results[results["raceId"].isin(circuit_races.index)]
        winner_dids = circuit_results[circuit_results["position"] == 1][["raceId", "driverId"]].values
        wdc_positions = []
        for rid, did in winner_dids:
            year = races.loc[rid, "year"]
            wdc_pos = wdc_final_positions[(wdc_final_positions["year"] == year) &
                                          (wdc_final_positions["driverId"] == did)]
            if wdc_pos.shape[0] > 0:
                wdc_positions.append(wdc_pos["position"].values[0])
        return np.mean(wdc_positions)

    def get_wdc_position(cid):
        # Gets the mean finishing position at this circuit for this WDC of the year
        circuit_races = races[races["circuitId"] == cid]
        circuit_results = results[results["raceId"].isin(circuit_races.index)]
        wdc_positions = []
        for year in circuit_races["year"].values:
            wdc_did = wdc_final_positions[(wdc_final_positions["year"] == year) &
                                          (wdc_final_positions["position"] == 1)]
            if wdc_did.shape[0] > 0:
                wdc_did = wdc_did["driverId"].values[0]
                wdc_position = circuit_results[circuit_results["driverId"] == wdc_did]
                if wdc_position.shape[0] > 0:
                    wdc_positions.append(wdc_position["positionOrder"].values[0])
        return np.mean(wdc_positions)

    source = pd.DataFrame(circuits.index.values, columns=["circuit_id"])
    source["winner_wdc_position"] = source["circuit_id"].apply(get_winner_wdc_position)
    source["wdc_position"] = source["circuit_id"].apply(get_wdc_position)
    source["circuit_name"] = source["circuit_id"].apply(get_circuit_name_custom)
    source["flag"] = source["circuit_name"].apply(lambda s: s[-2:])
    source["years"] = source["circuit_id"].apply(get_circuit_years)
    source["num_races"] = source["circuit_id"].apply(lambda cid: races[races["circuitId"] == cid].shape[0])
    source["size"] = source["num_races"].apply(get_circuit_size)

    palette = Turbo256
    n_circuits = source.shape[0]
    colors = []
    di = 180 / n_circuits
    i = 20
    for _ in range(n_circuits):
        colors.append(palette[int(i)])
        i += di
    source["color"] = colors

    upset_scatter = figure(title=u"Upset Plot",
                           x_axis_label="Avg. WDC position for winner at this circuit",
                           y_axis_label="Avg. Position for the WDC at this circuit",
                           x_range=Range1d(0, 10, bounds=(0, 60)),
                           y_range=Range1d(0, 20, bounds=(0, 60)),
                           plot_height=650)
    upset_scatter.xaxis.ticker = FixedTicker(ticks=np.arange(5, 61, 5).tolist() + [1])
    upset_scatter.yaxis.ticker = FixedTicker(ticks=np.arange(5, 61, 5).tolist() + [1])
    upset_scatter.xaxis.major_label_overrides = {i: int_to_ordinal(i) for i in range(1, 60)}
    upset_scatter.yaxis.major_label_overrides = {i: int_to_ordinal(i) for i in range(1, 60)}

    upset_scatter.scatter(x="winner_wdc_position", y="wdc_position", source=source, color="color", size="size")
    upset_scatter.line(x=[-60, 60], y=[-60, 60], color="white", line_alpha=0.5)

    marker_label_kwargs = dict(x="winner_wdc_position",
                               y="wdc_position",
                               level="glyph",
                               x_offset=0,
                               y_offset=0,
                               source=ColumnDataSource(source),
                               render_mode="canvas",
                               text_color="white",
                               text_font_size="10pt")
    labels = LabelSet(text="flag", **marker_label_kwargs)
    upset_scatter.add_layout(labels)

    label_kwargs = dict(render_mode="canvas",
                        text_color="white",
                        text_font_size="12pt")
    label1 = Label(x=0.5, y=19, text="Winner here does well in WDC", **label_kwargs)
    label2 = Label(x=0.5, y=18.2, text="Eventual WDC does poorly here", **label_kwargs)
    label3 = Label(x=5.5, y=1.5, text="Winner here does poorly in WDC", **label_kwargs)
    label4 = Label(x=5.5, y=0.7, text="Eventual WDC does well here", **label_kwargs)
    upset_scatter.add_layout(label1)
    upset_scatter.add_layout(label2)
    upset_scatter.add_layout(label3)
    upset_scatter.add_layout(label4)

    upset_scatter.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Circuit Name", "@circuit_name"),
        ("Number of races", "@num_races"),
        ("The winner at this circuit on average finishes in position", "@winner_wdc_position in the WDC"),
        ("The WDC finishes", "position @wdc_position at this circuit on average")
    ]))

    upset_scatter.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

    explanation = """The x axis is calculated by, for every circuit, finding the average World Drivers' Championship 
    position for the winner at this circuit.<br>
    The y axis is calculated by, for every circuit, finding the average finishing position for the World Driver's 
    Champion for that year at that circuit.<br>
    This means that dots far away from the origin represent circuits where the winner tends to finish low in the 
    championship and the (eventual) champion tends to finish low at this circuit, thus indicating that there are often 
    upsets. Dots near the origin indicate the opposite. Dots far from the origin near the x axis represent circuits 
    whose winner tends to do poorly in the WDC but the eventual World Champion does well here. Dots far from the 
    origin but near the y axis represent the opposite, or that the eventual World Champion does poorly here while 
    the winner at this circuit tends to do well in the WDC.<br>
    Dot size is calculated based on the number of races held at that circuit."""
    explanation = Div(text=explanation)

    return column([upset_scatter, explanation], sizing_mode="stretch_width")


def generate_spvfp_scatter():
    """
    Generates a scatter plot of starting position vs finish position.
    :return: SPvFP scatter plot layout
    """
    logging.info("Generating start pos vs finish pos scatter")

    def get_start_pos(cid):
        circuit_races = races[races["circuitId"] == cid]
        circuit_results = results[results["raceId"].isin(circuit_races.index)]
        return circuit_results["grid"].mean()

    def get_finish_pos(cid):
        circuit_races = races[races["circuitId"] == cid]
        circuit_results = results[results["raceId"].isin(circuit_races.index)]
        return circuit_results["positionOrder"].mean()

    source = pd.DataFrame(circuits.index.values, columns=["circuit_id"])
    source["start_pos"] = source["circuit_id"].apply(get_start_pos)
    source["finish_pos"] = source["circuit_id"].apply(get_finish_pos)
    source["circuit_name"] = source["circuit_id"].apply(get_circuit_name_custom)
    source["flag"] = source["circuit_name"].apply(lambda s: s[-2:])
    source["years"] = source["circuit_id"].apply(get_circuit_years)
    source["num_races"] = source["circuit_id"].apply(lambda cid: races[races["circuitId"] == cid].shape[0])
    source["size"] = source["num_races"].apply(get_circuit_size)

    palette = Turbo256
    n_circuits = source.shape[0]
    colors = []
    di = 180 / n_circuits
    i = 20
    for _ in range(n_circuits):
        colors.append(palette[int(i)])
        i += di
    source["color"] = colors

    spvfp_scatter = figure(title=u"Average Starting Position vs Average Finishing Position \u2014 includes DNFs",
                           x_axis_label="Avg. Starting Position at this Circuit",
                           y_axis_label="Avg. Finishing Position at this circuit",
                           x_range=Range1d(7, 20, bounds=(0, 60)),
                           y_range=Range1d(7, 20, bounds=(0, 60)),
                           plot_height=650)
    spvfp_scatter.xaxis.ticker = FixedTicker(ticks=np.arange(5, 61, 5).tolist() + [1])
    spvfp_scatter.yaxis.ticker = FixedTicker(ticks=np.arange(5, 61, 5).tolist() + [1])
    spvfp_scatter.xaxis.major_label_overrides = {i: int_to_ordinal(i) for i in range(1, 60)}
    spvfp_scatter.yaxis.major_label_overrides = {i: int_to_ordinal(i) for i in range(1, 60)}
    subtitle = "Dot size is calculated based on the number of races held at that circuit"
    spvfp_scatter.add_layout(Title(text=subtitle, text_font_style="italic"), "above")

    spvfp_scatter.scatter(x="start_pos", y="finish_pos", source=source, color="color", size="size")
    spvfp_scatter.line(x=[-60, 60], y=[-60, 60], color="white", line_alpha=0.5)

    label_kwargs = dict(render_mode="canvas",
                        text_color="white",
                        text_font_size="12pt")
    label1 = Label(x=16, y=9, text="Drivers tend to finish higher than started", **label_kwargs)
    label2 = Label(x=8, y=19, text="Drivers tend to finish lower than started", **label_kwargs)
    spvfp_scatter.add_layout(label1)
    spvfp_scatter.add_layout(label2)

    spvfp_scatter.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Circuit Name", "@circuit_name"),
        ("Number of races", "@num_races"),
        ("Average starting position", "@start_pos"),
        ("Average finishing position", "@finish_pos")
    ]))

    spvfp_scatter.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

    return spvfp_scatter


def generate_num_overtakes_bar():
    """
    Generates number of overtakes bar plot. No, this is not just mirroring what the DNF percent bar chart says.
    :return: Num overtakes bar plot layout
    """
    logging.info("Generating overtakes bar plot")

    def get_num_overtakes(cid):
        circuit_races = races[races["circuitId"] == cid]
        return circuit_races["num_overtakes"].mean()

    source = pd.DataFrame(circuits.index.values, columns=["circuit_id"])
    source["avg_num_overtakes"] = source["circuit_id"].apply(get_num_overtakes)
    source = source[(source["avg_num_overtakes"].notna()) & (source["avg_num_overtakes"] > 0)]
    source["circuit_name"] = source["circuit_id"].apply(get_circuit_name_custom)
    source["avg_num_overtakes_str"] = source["avg_num_overtakes"].apply(lambda x: str(round(x, 1)))
    source = source.sort_values(by="avg_num_overtakes", ascending=False)

    overtakes_bar = figure(
        title=u"Average Number of Overtakes* per Race (1996 onward)",
        y_axis_label="Avg. Num. Overtakes",
        x_range=source["circuit_name"].unique(),
        y_range=Range1d(0, 150, bounds=(0, 150)),
        plot_height=600,
        toolbar_location=None,
        tools="",
        tooltips="Circuit: @circuit_name<br>"
                 "Average Number of Overtakes: @avg_num_overtakes_str"
    )
    overtakes_bar.xaxis.major_label_orientation = math.pi / 2

    palette = Turbo256
    n_circuits = source.shape[0]
    colors = []
    di = 230 / n_circuits
    i = 20
    for _ in range(n_circuits):
        colors.append(palette[int(i)])
        i += di
    source["color"] = colors

    overtakes_bar.vbar(x="circuit_name", top="avg_num_overtakes", source=source, width=0.8, color="color")

    explanation = """Overtakes here are considered as a position change that last at least a lap. This means that if 
    two drivers swap places then swap back in the same lap, it is not counted, but if they swap one lap then swap 
    back the next, it is counted.<br>
    Furthermore, overtakes from DNFs are counted.<br>
    Though this overtake metric does not precisely match the official definition of an overtaking move, the relative 
    positions of circuits still is useful."""
    explanation = Div(text=explanation)

    return row([overtakes_bar, explanation], sizing_mode="stretch_width")


def generate_sc_laps_bar():
    """
    Generates number of safety car laps bar plot.
    :return: Safety car bar plot layout.
    """
    logging.info("Generating safety car laps bar plot")

    source = pd.DataFrame(races.groupby("circuitId")["SCLaps"].mean())
    source = source[source["SCLaps"].notna()]
    source["circuit_name"] = source.index.map(get_circuit_name_custom)
    source = source.sort_values(by="SCLaps", ascending=False)

    sc_bar = figure(
        title=u"Average Number of Safety Car Laps at Each Circuit (2009-2017)",
        y_axis_label="Avg. Num. Safety Car Laps",
        x_range=source["circuit_name"].unique(),
        y_range=Range1d(0, 15, bounds=(0, 15)),
        plot_height=600,
        toolbar_location=None,
        tools="",
        tooltips="Circuit: @circuit_name<br>"
                 "Avg. Safety Car Laps: @SCLaps"
    )
    sc_bar.xaxis.major_label_orientation = math.pi / 2

    palette = Turbo256
    n_circuits = source.shape[0]
    colors = []
    di = 230 / n_circuits
    i = 20
    for _ in range(n_circuits):
        colors.append(palette[int(i)])
        i += di
    source["color"] = colors

    sc_bar.vbar(x="circuit_name", top="SCLaps", source=source, width=0.8, color="color")

    return sc_bar


def generate_weather_bar():
    """
    Generates a bar plot showing weather at circuits.
    :return: Weather bar plot layout
    """
    source = races.groupby("circuitId")["weather"].value_counts(normalize=True)

    cids = set()
    for idx, _ in source.iteritems():
        cids.add(idx[0])

    circuit_names = []
    drys = []
    varieds = []
    wets = []
    weathers = ["Dry", "Varied", "Wet"]
    colors = ["green", "orange", "red"]

    for cid in cids:
        dry = 0
        varied = 0
        wet = 0
        counts_row = source.loc[cid]
        if "dry" in counts_row:
            dry = counts_row["dry"]
        if "varied" in counts_row:
            varied = counts_row["varied"]
        if "wet" in counts_row:
            wet = counts_row["wet"]
        dry *= 100
        varied *= 100
        wet *= 100
        drys.append(dry)
        varieds.append(varied)
        wets.append(wet)
        circuit_names.append(get_circuit_name_custom(cid))

    data = {
        "circuit_name": circuit_names,
        "Dry": drys,
        "Varied": varieds,
        "Wet": wets
    }

    weather_bar = figure(
        title=u"Circuit Weather (2009-2017)",
        y_axis_label="Percentage",
        x_range=circuit_names,
        y_range=Range1d(0, 100, bounds=(0, 100)),
        plot_height=600,
        toolbar_location=None,
        tools="",
        tooltips="Circuit: @circuit_name<br>"
                 "$name @circuit_name: @$name%"
    )
    weather_bar.yaxis.formatter = PrintfTickFormatter(format="%s%%")
    weather_bar.xaxis.major_label_orientation = math.pi / 2

    weather_bar.vbar_stack(weathers, x="circuit_name", width=0.8, color=colors, source=data,
                           legend_label=weathers)

    weather_bar.legend.location = "top_left"
    weather_bar.legend.orientation = "horizontal"

    return weather_bar


def generate_circuits_table():
    """
    Generates a table with information on every circuit.
    :return: Circuits table layout
    """
    source = pd.DataFrame(columns=["circuit_name", "location", "num_races", "years", "top_driver", "top_constructor"])
    for cid, circuit_row in circuits.iterrows():
        circuit_name = get_circuit_name(cid)
        location = circuit_row["location"] + ", " + circuit_row["country"]
        circuit_races = races[races["circuitId"] == cid]
        num_races = circuit_races.shape[0]
        years = circuit_races["year"].unique()
        years.sort()
        years = rounds_to_str(years)
        circuit_winners = results[(results["raceId"].isin(circuit_races.index)) & (results["position"] == 1)]
        driver_winners = circuit_winners["driverId"].value_counts()
        top_num_wins = driver_winners.iloc[0]
        top_driver_winners = []
        for did, num_wins in driver_winners.iteritems():
            if num_wins == top_num_wins:
                top_driver_winners.append(get_driver_name(did) + " (" + str(num_wins) + " wins)")
            else:
                break
        top_driver_winners = ", ".join(top_driver_winners)

        constructor_winners = circuit_winners["constructorId"].value_counts()
        top_num_wins = constructor_winners.iloc[0]
        top_constructor_winners = []
        for constructor_id, num_wins in constructor_winners.iteritems():
            if num_wins == top_num_wins:
                top_constructor_winners.append(get_constructor_name(constructor_id) + " (" + str(num_wins) + " wins)")
            else:
                break
        top_constructor_winners = ", ".join(top_constructor_winners)

        source = source.append({
            "circuit_name": circuit_name,
            "location": location,
            "num_races": num_races,
            "years": years,
            "top_driver": top_driver_winners,
            "top_constructor": top_constructor_winners
        }, ignore_index=True)
    source = source.sort_values(by="num_races", ascending=False)

    circuits_columns = [
        TableColumn(field="circuit_name", title="Circuit Name", width=150),
        TableColumn(field="location", title="Location", width=100),
        TableColumn(field="num_races", title="Num. Races", width=50),
        TableColumn(field="years", title="Years", width=130),
        TableColumn(field="top_driver", title="Top Winner(s) (Driver)", width=275),
        TableColumn(field="top_constructor", title="Top Winner(s) (Constructor)", width=200),
    ]

    circuits_table = DataTable(source=ColumnDataSource(data=source), columns=circuits_columns, index_position=None)

    title = Div(text="<h2><b>All Circuits</b></h2>")

    return column([title, circuits_table], sizing_mode="stretch_width")


def get_circuit_name_custom(cid):
    circuit_name = get_circuit_name(cid)
    flag = circuit_name[0:2]
    circuit_name = circuit_name[3:] + " " + flag
    if "Autódromo Internacional Nelson Piquet" in circuit_name:
        circuit_name = circuit_name.replace("Autódromo Internacional Nelson Piquet", "Nelson Piquet (Jacarepaguá)")
    return circuit_name


def get_circuit_years(cid):
    years = races[races["circuitId"] == cid]["year"].unique()
    years.sort()
    return rounds_to_str(years)


def get_circuit_size(n_races):
    return math.pow(n_races, 0.5) + 2
