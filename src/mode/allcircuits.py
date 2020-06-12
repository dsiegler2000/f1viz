import logging
import math
from bokeh.layouts import column
from bokeh.models import Range1d, Div, NumeralTickFormatter, Title, Label, LabelSet, ColumnDataSource, \
    DatetimeTickFormatter, HoverTool
from bokeh.palettes import Turbo256
from bokeh.plotting import figure
import pandas as pd
import numpy as np
from data_loading.data_loader import load_circuits, load_races, load_results, load_fastest_lap_data, load_pit_stops, \
    load_driver_standings, load_wdc_final_positions
from utils import get_circuit_name, rounds_to_str, get_status_classification, millis_to_str, DATETIME_TICK_KWARGS

circuits = load_circuits()
circuits = circuits.drop(72)  # Drop Port Imperial Street Circuit as it was never raced on
circuits = circuits.drop(74)  # Drop Hanoi Street Circuit for now
races = load_races()
results = load_results()
fastest_lap_data = load_fastest_lap_data()
pit_stop_data = load_pit_stops()
driver_standings = load_driver_standings()
wdc_final_positions = load_wdc_final_positions()

# TODO make the upsets scatter, try as 2 bar plots if it doesn't work potentially (see Trello)


def get_layout(**kwargs):
    years = races["year"].unique()
    years.sort()

    num_races_bar_chart = generate_num_races_bar()

    dnf_bar_chart = generate_dnf_bar()

    mspmfp_bar_chart = generate_mspmfp_bar()

    countries_bar_chart = generate_countries_bar()

    rating_bar_chart = generate_rating_bar()

    avg_lap_time_bar_chart = generate_avg_lap_time_bar()

    pit_stop_bar_chart = generate_pit_stop_bar()

    upset_scatter = generate_upset_scatter(years)

    header = Div(text=u"<h2><b>All Circuits \u2014 Some Stats on All Circuits that have held F1 Races")

    middle_spacer = Div()
    layout = column([
        header,
        num_races_bar_chart, middle_spacer,
        dnf_bar_chart, middle_spacer,
        countries_bar_chart, middle_spacer,
        mspmfp_bar_chart, middle_spacer,
        rating_bar_chart, middle_spacer,
        avg_lap_time_bar_chart, middle_spacer,
        pit_stop_bar_chart, middle_spacer,
        upset_scatter, middle_spacer
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


def generate_upset_scatter(years):
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
                    # TODO try to change this to just "position"
                    wdc_positions.append(wdc_position["positionOrder"].shape[0])
        return np.mean(wdc_positions)

    source = pd.DataFrame(circuits.index.values, columns=["circuit_id"])
    source["winner_wdc_position"] = source["circuit_id"].apply(get_winner_wdc_position)
    source["wdc_position"] = source["circuit_id"].apply(get_wdc_position)
    source["circuit_name"] = source["circuit_id"].apply(get_circuit_name_custom)

    # TODO
    #  hover tooltip that has circuit name, each axis data, number of races held there
    #  axes, axes labels, titel, explanation, potentially even label every dot w/ circuit name
    #  draw the y=x line (I think that's the relevant one at least)
    #  crosshair
    #  label regions
    #  make dots bigger
    #  potentially determine size or shape based on number of races there
    #  see the impact of changing from "positionOrder" to "position" (see above)
    #  check other scatters for stuff

    palette = Turbo256
    n_circuits = source.shape[0]
    colors = []
    di = 230 / n_circuits
    i = 20
    for _ in range(n_circuits):
        colors.append(palette[int(i)])
        i += di
    source["color"] = colors

    upset_scatter = figure()

    upset_scatter.scatter(x="winner_wdc_position", y="wdc_position", source=source, color="color")

    tooltips = [
        ("Circuit Name", "@circuit_name")
    ]
    upset_scatter.add_tools(HoverTool(show_arrow=False, tooltips=tooltips))

    return upset_scatter


def get_circuit_name_custom(cid):
    circuit_name = get_circuit_name(cid)
    flag = circuit_name[0:2]
    circuit_name = circuit_name[3:] + " " + flag
    if "Autódromo Internacional Nelson Piquet" in circuit_name:
        circuit_name = circuit_name.replace("Autódromo Internacional Nelson Piquet", "Nelson Piquet (Jacarepaguá)")
    return circuit_name
