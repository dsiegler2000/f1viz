import logging
from collections import defaultdict
import numpy as np
from bokeh.layouts import column, row
from bokeh.models import Div, HoverTool, LegendItem, Legend, CrosshairTool, Range1d, DataRange1d, LinearAxis, \
    DatetimeTickFormatter, Spacer, Span, ColumnDataSource, TableColumn, DataTable, NumeralTickFormatter, Title, Label
from bokeh.plotting import figure
from data_loading.data_loader import load_circuits, load_fastest_lap_data, load_races, load_qualifying, load_results, \
    load_driver_standings
from mode import driver
from utils import get_circuit_name, get_driver_name, millis_to_str, DATETIME_TICK_KWARGS, PLOT_BACKGROUND_COLOR, \
    get_constructor_name, vdivider, plot_image_url, rounds_to_str, get_status_classification
import pandas as pd

circuits = load_circuits()
fastest_lap_data = load_fastest_lap_data()
races = load_races()
qualifying = load_qualifying()
results = load_results()
driver_standings = load_driver_standings()

# TODO this really needs a box to check for "generate SP v FP plot" and "generate MLTR vs FP plot" for efficiency


def get_layout(circuit_id=-1, download_image=True, **kwargs):
    circuit_races = races[races["circuitId"] == circuit_id]
    circuit_rids = circuit_races.index
    circuit_years = sorted(circuit_races["year"].values.tolist())
    circuit_fastest_lap_data = fastest_lap_data[fastest_lap_data["raceId"].isin(circuit_rids)]
    circuit_quali = qualifying[qualifying["raceId"].isin(circuit_rids)]
    circuit_results = results[results["raceId"].isin(circuit_rids)]
    circuit_driver_standings = driver_standings[driver_standings["raceId"].isin(circuit_rids)]

    logging.info(f"Generating layout for mode CIRCUIT in circuit, circuit_id={circuit_id}")

    # Generate times plot
    times_plot = generate_times_plot(circuit_years, circuit_quali, circuit_fastest_lap_data, circuit_races,
                                     circuit_results, circuit_id)

    # Generate DNF plot
    dnf_plot = generate_dnf_plot(circuit_years, circuit_results, circuit_races, circuit_id)

    # Generate starting position minus finish position plot
    spmfp_plot = generate_spmfp_plot(circuit_years, circuit_races, circuit_results)

    # Start pos vs finish pos scatter
    spvfp_scatter = generate_spvfp_scatter(circuit_results, circuit_races, circuit_driver_standings)

    # Mean lap time rank vs finish pos scatter
    mltr_fp_scatter = generate_mltr_fp_scatter(circuit_results, circuit_races, circuit_driver_standings)

    # Generate results table
    circuit_results_table = generate_circuit_results_table(circuit_years, circuit_races, circuit_results,
                                                           circuit_quali, circuit_fastest_lap_data)

    # Circuit stats
    circuit_stats = generate_stats_layout(circuit_id, circuit_years, circuit_fastest_lap_data, circuit_results,
                                          circuit_races, download_image=download_image)

    # Winner's table
    winners_table = generate_winners_table(circuit_years, circuit_results, circuit_races)

    # Header
    circuit_name = get_circuit_name(circuit_id)
    header = Div(text=f"<h2><b>{circuit_name}</b></h2><br>")

    middle_spacer = Spacer(width=5, background=PLOT_BACKGROUND_COLOR)
    layout = column([
        header,
        times_plot, middle_spacer,
        dnf_plot, middle_spacer,
        spmfp_plot, middle_spacer,
        row([spvfp_scatter, mltr_fp_scatter], sizing_mode="stretch_width"),
        circuit_results_table,
        circuit_stats,
        winners_table
    ], sizing_mode="stretch_width")

    logging.info("Finished generating layout for mode CIRCUIT")

    return layout


def generate_spvfp_scatter(circuit_results, circuit_races, circuit_driver_standings):
    """
    Start position vs finish position scatter
    :param circuit_results: Circuit results
    :param circuit_races: Circuit races
    :param circuit_driver_standings: Circuit driver standings
    :return: Scatter plot layout
    """
    return driver.generate_spvfp_scatter(circuit_results, circuit_races, circuit_driver_standings, color_drivers=True)


def generate_mltr_fp_scatter(circuit_results, circuit_races, circuit_driver_standings):
    """
    Mean lap time rank vs finish position scatter
    :param circuit_results: Circuit results
    :param circuit_races: Circuit races
    :param circuit_driver_standings: Circuit driver standings
    :return: Mean lap time vs finish pos scatter layout
    """
    return driver.generate_mltr_fp_scatter(circuit_results, circuit_races, circuit_driver_standings, color_drivers=True)


def generate_dnf_plot(circuit_years, circuit_results, circuit_races, circuit_id):
    """
    Plots number of races, number of DNFs, and DNF percent for that year on the same plot. (2 different axes).
    :param circuit_years: Circuit years
    :param circuit_results: Circuit results
    :param circuit_races: Circuit races
    :param circuit_id: Circuit ID
    :return: Plot layout
    """
    # TODO refactor to use existing method
    logging.info("Generating dnf plot")
    if len(circuit_years) == 0:
        return Div()
    source = pd.DataFrame(columns=["n_races", "year", "n_drivers",
                                   "dnf_pct", "dnfs", "dnf_pct_str",
                                   "total_dnf_pct", "total_dnfs", "total_dnf_pct_str"])
    n_races = 0
    total_dnfs = 0
    total_drivers = 0
    for year in circuit_years:
        year_race = circuit_races[circuit_races["year"] == year]
        if year_race.shape[0] == 0:
            continue
        rid = year_race.index.values[0]
        year_results = circuit_results[circuit_results["raceId"] == rid]
        num_dnfs = year_results["position"].isna().sum()
        num_drivers = year_results.shape[0]
        total_dnfs += num_dnfs
        total_drivers += num_drivers
        if num_drivers > 0:
            dnf_pct = num_dnfs / num_drivers
            total_dnf_pct = total_dnfs / total_drivers
            n_races += 1
            source = source.append({
                "n_races": n_races,
                "n_drivers": num_drivers,
                "year": year,
                "dnf_pct": dnf_pct,
                "dnfs": num_dnfs,
                "dnf_pct_str": str(round(100 * dnf_pct, 1)) + "%",
                "total_dnf_pct": total_dnf_pct,
                "total_dnfs": total_dnfs,
                "total_dnf_pct_str": str(round(100 * total_dnf_pct, 1)) + "%",
            }, ignore_index=True)

    circuit_name = get_circuit_name(circuit_id)
    min_year = min(circuit_years)
    max_year = max(circuit_years)
    max_drivers = source["n_drivers"].max()
    if max_drivers == 0:
        return Div()
    dnf_plot = figure(
        title=u"Number of DNFs \u2014 " + circuit_name,
        y_axis_label="",
        x_axis_label="Year",
        x_range=Range1d(min_year, max_year, bounds=(min_year, max_year + 3)),
        tools="pan,xbox_zoom,xwheel_zoom,reset,box_zoom,wheel_zoom,save",
        y_range=Range1d(0, max_drivers, bounds=(-1000, 1000))
    )

    subtitle = 'Year DNFs refers to the number/percent of DNFs for that year, Total DNFs refers to all DNFs up to ' \
               'that point in time'
    dnf_plot.add_layout(Title(text=subtitle, text_font_style="italic"), "above")

    max_dnf_pct = max(source["dnf_pct"].max(), source["total_dnf_pct"].max())
    if max_dnf_pct > 0:
        k = max_drivers / max_dnf_pct
    else:
        k = 1
    source["dnf_pct_scaled"] = k * source["dnf_pct"]
    source["total_dnf_pct_scaled"] = k * source["total_dnf_pct"]

    # Other y axis
    y_range = Range1d(start=0, end=max_dnf_pct, bounds=(-0.02, 1000))
    dnf_plot.extra_y_ranges = {"percent_range": y_range}
    axis = LinearAxis(y_range_name="percent_range")
    axis.formatter = NumeralTickFormatter(format="0.0%")
    dnf_plot.add_layout(axis, "right")

    kwargs = {
        "x": "year",
        "line_width": 2,
        "line_alpha": 0.7,
        "source": source,
        "muted_alpha": 0.05
    }

    races_line = dnf_plot.line(y="n_races", color="white", **kwargs)
    drivers_line = dnf_plot.line(y="n_drivers", color="yellow", **kwargs)
    dnfs_line = dnf_plot.line(y="dnfs", color="aqua", **kwargs)
    dnf_pct_line = dnf_plot.line(y="dnf_pct_scaled", color="aqua", line_dash="dashed", **kwargs)
    total_dnfs_line = dnf_plot.line(y="total_dnfs", color="orange", **kwargs)
    total_dnf_pct_line = dnf_plot.line(y="total_dnf_pct_scaled", color="orange", line_dash="dashed", **kwargs)

    legend = [LegendItem(label="Number of Races", renderers=[races_line]),
              LegendItem(label="Number of Drivers", renderers=[drivers_line]),
              LegendItem(label="Year DNFs", renderers=[dnfs_line]),
              LegendItem(label="Year DNF Pct.", renderers=[dnf_pct_line]),
              LegendItem(label="Total DNFs", renderers=[total_dnfs_line]),
              LegendItem(label="Total DNF Pct.", renderers=[total_dnf_pct_line])]

    legend = Legend(items=legend, location="top_right", glyph_height=15, spacing=2, inactive_fill_color="gray")
    dnf_plot.add_layout(legend, "right")
    dnf_plot.legend.click_policy = "mute"
    dnf_plot.legend.label_text_font_size = "12pt"

    # Hover tooltip
    dnf_plot.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Number of Races", "@n_races"),
        ("Number of Drivers", "@n_drivers"),
        ("Year Num. DNFs", "@dnfs (@dnf_pct_str)"),
        ("Total Num. DNFs", "@total_dnfs (@total_dnf_pct_str)"),
        ("Year", "@year"),
    ]))

    # Crosshair tooltip
    dnf_plot.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

    return dnf_plot


def generate_times_plot(circuit_years, circuit_quali, circuit_fastest_lap_data, circuit_races, circuit_results,
                        circuit_id):
    """
    Plot quali, fastest lap, and average lap times vs year along with rating vs year
    :param circuit_years: Circuit years
    :param circuit_quali: Circuit quali
    :param circuit_fastest_lap_data: Circuit fastest lap data
    :param circuit_races: Circuit races
    :param circuit_results: Circuit results
    :param circuit_id: Circuit ID
    :return: Times plot layout
    """
    logging.info("Generating times plot")
    if circuit_quali.shape[0] == 0 and (circuit_fastest_lap_data.shape[0] == 0 or
                                        circuit_fastest_lap_data["avg_lap_time_millis"].isna().sum() ==
                                        circuit_fastest_lap_data["avg_lap_time_millis"].shape[0]):
        return Div(text="")
    source = pd.DataFrame(columns=["year",
                                   "quali_time", "quali_time_str",
                                   "fastest_lap_time", "fastest_lap_str",
                                   "avg_lap_time", "avg_lap_str",
                                   "rating"])

    for year in circuit_years:
        race = circuit_races[circuit_races["year"] == year]
        rid = race.index.values[0]

        # Qualifying
        year_quali = circuit_quali[circuit_quali["raceId"] == rid]
        quali_times = year_quali["q1"].append(year_quali["q2"].append(year_quali["q3"]))
        if quali_times.shape[0] == 0 or np.isnan(quali_times.idxmin()):
            quali_millis = np.nan
            quali_str = ""
        else:
            idxmin = int(quali_times.idxmin())
            quali_name = get_driver_name(year_quali.loc[idxmin, "driverId"])
            quali_millis = int(quali_times.loc[idxmin].min())
            quali_str = millis_to_str(quali_millis) + " by " + quali_name

        # Fastest and average lap
        year_fastest_lap_data = circuit_fastest_lap_data[circuit_fastest_lap_data["raceId"] == rid]
        if year_fastest_lap_data.shape[0] == 0:
            avg_lap_millis = np.nan
            avg_lap_str = ""
        else:
            avg_lap_millis = int(year_fastest_lap_data["avg_lap_time_millis"].mean())
            avg_lap_str = millis_to_str(avg_lap_millis)
        if year_fastest_lap_data.shape[0] == 0 or \
                year_fastest_lap_data["fastest_lap_time_millis"].isna().sum() == year_fastest_lap_data.shape[0]:
            fastest_lap_millis = np.nan
            fastest_lap_str = ""
        else:
            idxmin = int(year_fastest_lap_data["fastest_lap_time_millis"].idxmin())
            fastest_lap_name = get_driver_name(year_fastest_lap_data.loc[idxmin, "driver_id"])
            fastest_lap_millis = int(year_fastest_lap_data.loc[idxmin, "fastest_lap_time_millis"])
            fastest_lap_str = millis_to_str(fastest_lap_millis) + " by " + fastest_lap_name

        source = source.append({
            "year": year,
            "quali_time": quali_millis,
            "quali_time_str": quali_str,
            "fastest_lap_time": fastest_lap_millis,
            "fastest_lap_str": fastest_lap_str,
            "avg_lap_time": avg_lap_millis,
            "avg_lap_str": avg_lap_str,
            "rating": race["rating"].values[0],
        }, ignore_index=True)

    circuit_name = get_circuit_name(circuit_id)
    min_time = min(source["fastest_lap_time"].min(), source["avg_lap_time"].min(), source["quali_time"].min()) - 5000
    max_time = max(source["fastest_lap_time"].max(), source["avg_lap_time"].max(), source["quali_time"].max()) + 5000
    start = pd.to_datetime(min_time, unit="ms")
    end = pd.to_datetime(max_time, unit="ms")
    if pd.isna(start) or pd.isna(end):
        min_time = source["avg_lap_time"].min() - 5000
        max_time = source["avg_lap_time"].max() + 5000
        start = pd.to_datetime(min_time, unit="ms")
        end = pd.to_datetime(max_time, unit="ms")

    # Scale rating so that a 0=min_time, 10=max_time
    source["rating_scaled"] = (max_time - min_time) * (source["rating"] / 10) + min_time
    source["rating"] = source["rating"].fillna("")

    min_year = np.min(circuit_years)
    max_year = np.max(circuit_years)
    if min_year == max_year:
        min_year -= 0.01

    times_plot = figure(
        title=u"Qualifying, Fastest, Average, and Winning Times for " + circuit_name +
              " \u2014 Some data may be missing, zoom for more detail",
        x_axis_label="Year",
        y_axis_label="Lap Time",
        y_range=DataRange1d(start=start, end=end, bounds=(start, end)),
        x_range=Range1d(min_year, max_year, bounds=(min_year, max_year + 3)),
        tools="pan,xbox_zoom,xwheel_zoom,reset,box_zoom,wheel_zoom,save"
    )

    times_plot.yaxis.formatter = DatetimeTickFormatter(**DATETIME_TICK_KWARGS)
    column_source = ColumnDataSource(data=source)
    kwargs = {
        "x": "year",
        "source": column_source,
        "line_width": 2,
        "muted_alpha": 0.05
    }

    avg_lap_time_line = times_plot.line(y="avg_lap_time", line_color="white", **kwargs)
    legend_items = [
        LegendItem(label="Average Race Lap", renderers=[avg_lap_time_line]),
    ]
    tooltips = [
        ("Year", "@year"),
        ("Average Lap Time", "@avg_lap_str"),
    ]

    if source["quali_time"].isna().sum() < source.shape[0]:
        quali_time_line = times_plot.line(y="quali_time", line_color="red", **kwargs)
        legend_items.append(LegendItem(label="Qualifying Fastest", renderers=[quali_time_line]))
        tooltips.append(("Qualifying Lap Time", "@quali_time_str"))

    if source["fastest_lap_time"].isna().sum() < source.shape[0]:
        fastest_lap_time_line = times_plot.line(y="fastest_lap_time", line_color="yellow", **kwargs)
        legend_items.append(LegendItem(label="Fastest Race Lap", renderers=[fastest_lap_time_line]))
        tooltips.append(("Fastest Lap Time", "@fastest_lap_str"))

    # Add rating and other axis
    if source["rating"].replace("", np.nan).isna().sum() < source.shape[0]:
        rating_line = times_plot.line(y="rating_scaled", line_color="green", line_alpha=0.9, name="rating_line",
                                      **kwargs)
        legend_items.append(LegendItem(label="Average Rating", renderers=[rating_line]))
        tooltips.append(("Rating", "@rating"))
        y_range = Range1d(start=0, end=10, bounds=(0, 10))
        times_plot.extra_y_ranges = {"rating_range": y_range}
        axis = LinearAxis(y_range_name="rating_range", axis_label="Rating")
        times_plot.add_layout(axis, "right")

        def update_rating_axis():
            def dt_to_millis(t):
                if isinstance(t, float) or isinstance(t, int):
                    return t
                return t.microsecond / 1000 + t.second * 1000 + t.minute * 1000 * 60
            max_time = dt_to_millis(times_plot.y_range.end)
            min_time = dt_to_millis(times_plot.y_range.start)
            new_rating = (max_time - min_time) * (source["rating"].replace("", np.nan).astype(float) / 10) + min_time
            column_source.patch({
                "rating_scaled": [(slice(new_rating.shape[0]), new_rating)]
            })
            times_plot.extra_y_ranges.update({"rating_range": Range1d(start=0, end=10, bounds=(0, 10))})

        times_plot.y_range.on_change("start", lambda attr, old, new: update_rating_axis())
        times_plot.y_range.on_change("end", lambda attr, old, new: update_rating_axis())

    # Legend
    legend = Legend(items=legend_items, location="top_right", glyph_height=15, spacing=2, inactive_fill_color="gray")
    times_plot.add_layout(legend, "right")
    times_plot.legend.click_policy = "mute"
    times_plot.legend.label_text_font_size = "12pt"  # The default font size

    # Hover tooltip
    times_plot.add_tools(HoverTool(show_arrow=False, tooltips=tooltips))

    # Crosshair
    times_plot.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

    return times_plot


def generate_spmfp_plot(circuit_years, circuit_races, circuit_results):
    """
    Plot mean of starting position - finish position and num DNFs vs time, with horizontal line showing average
    :param circuit_years: Circuit years
    :param circuit_races: Circuit races
    :param circuit_results: Circuit results
    :return: SPmFP plot layout
    """
    logging.info("Generating SPMFP plot")
    source = pd.DataFrame(columns=["year", "msp", "mfp", "mspmfp", "dnf"])
    for year in circuit_years:
        race = circuit_races[circuit_races["year"] == year]
        rid = race.index.values[0]
        results = circuit_results[circuit_results["raceId"] == rid]
        finishing_pos = pd.to_numeric(results["positionText"], errors="coerce")
        mspmfp = (results["grid"] - finishing_pos).mean()
        dnf = results.shape[0] - results["positionText"].str.isnumeric().sum()
        source = source.append({
            "year": year,
            "msp": round(results["grid"].mean(), 1),
            "mfp": finishing_pos.mean(),
            "mspmfp": mspmfp,
            "dnf": dnf
        }, ignore_index=True)

    min_year = source["year"].min()
    max_year = source["year"].max()
    min_y = min(source["dnf"].min(), source["mspmfp"].min(), 1.5)
    max_y = max(source["dnf"].max(), source["mspmfp"].max())
    if min_year == max_year:
        min_year -= 1
        max_year += 1
    spfp_plot = figure(
        title=u"Average Start Position minus Finish Position \u2014 How many places do drivers make up on average?",
        x_axis_label="Year",
        y_axis_label="Average start position minus finish position",
        x_range=Range1d(min_year, max_year, bounds=(min_year, max_year + 3)),
        y_range=Range1d(min(min_y - 2, -1), max_y + 2, bounds=(min(min_y - 2, -1), max_y)),
        tools="pan,xbox_zoom,xwheel_zoom,reset,box_zoom,wheel_zoom,save"
    )

    kwargs = {
        "x": "year",
        "source": source,
        "muted_alpha": 0.05
    }
    mspmfp_line = spfp_plot.line(y="mspmfp", line_width=2, color="white", **kwargs)
    dnf_line = spfp_plot.line(y="dnf", line_width=1.9, color="orange", line_alpha=0.8, **kwargs)

    # Mean lines
    kwargs = {
        "x": [1950, 2100],
        "line_alpha": 0.5,
        "line_width": 2,
        "line_dash": "dashed",
        "muted_alpha": 0.08
    }
    mspmfp_mean_line = spfp_plot.line(y=[source["mspmfp"].mean()] * 2, line_color="white", **kwargs)
    dnf_mean_line = spfp_plot.line(y=[source["dnf"].mean()] * 2, line_color="orange", **kwargs)

    # Zero line
    spfp_plot.add_layout(Span(line_color="white", location=0, dimension="width", line_alpha=0.3, line_width=1.5))

    # Legend
    legend_items = [
        LegendItem(label="Average Start - Finish Pos.", renderers=[mspmfp_line, mspmfp_mean_line]),
        LegendItem(label="Number of DNFs", renderers=[dnf_line, dnf_mean_line]),
    ]
    legend = Legend(items=legend_items, location="top_right", glyph_height=15, spacing=2, inactive_fill_color="gray")
    spfp_plot.add_layout(legend, "right")
    spfp_plot.legend.click_policy = "mute"
    spfp_plot.legend.label_text_font_size = "12pt"  # The default font size

    # Labels
    text_label_kwargs = dict(x=min_year + 0.5,
                             render_mode="canvas",
                             text_color="white",
                             text_font_size="12pt",
                             border_line_color="white",
                             border_line_alpha=0.7)
    label1 = Label(y=-0.9, text=" Finish lower than started ", **text_label_kwargs)
    label2 = Label(y=0.4, text=" Finish higher than start ", **text_label_kwargs)
    spfp_plot.add_layout(label1)
    spfp_plot.add_layout(label2)

    # Hover tooltip
    spfp_plot.add_tools(HoverTool(show_arrow=False, tooltips=[
        ("Year", "@year"),
        ("Mean Starting Position", "@msp"),
        ("Mean Finishing Position", "@mfp"),
        ("Mean Start minus Finish Position", "@mspmfp"),
        ("Number of DNFs", "@dnf"),
    ]))

    # Crosshair
    spfp_plot.add_tools(CrosshairTool(line_color="white", line_alpha=0.6))

    return spfp_plot


def generate_circuit_results_table(circuit_years, circuit_races, circuit_results, circuit_qualifying,
                                   circuit_fastest_lap_data):
    """
    Show results for every race at that circuit, including polesetting and time, winner and time, 2nd and 3rd, and
    fastest lap if that data is available
    :param circuit_years: Circuit years
    :param circuit_races: Circuit races
    :param circuit_results: Circuit results
    :param circuit_qualifying: Circuit qualifying
    :param circuit_fastest_lap_data: Circuit fastest lap data
    :return: Table layout
    """
    logging.info("Generating circuit results table")
    source = pd.DataFrame(columns=["year", "laps", "pole", "fastest_lap", "p1", "p2", "p3"])
    for year in circuit_years:
        race = circuit_races[circuit_races["year"] == year]
        rid = race.index.values[0]
        results = circuit_results[circuit_results["raceId"] == rid]

        # Qualifying
        if results.shape[0] == 0:
            continue
        pole_row = results[results["grid"] == 1].iloc[0]
        pole_did = pole_row["driverId"]
        pole_name = get_driver_name(pole_did)
        quali = circuit_qualifying[(circuit_qualifying["raceId"] == rid) & (circuit_qualifying["driverId"] == pole_did)]
        quali = quali
        pole_time = ""
        if quali.shape[0] > 0:
            quali = quali.iloc[0]
            if not np.isnan(quali["q1"]):
                pole_time = millis_to_str(quali["q1"])
            if not np.isnan(quali["q2"]):
                pole_time = millis_to_str(quali["q2"])
            if not np.isnan(quali["q3"]):
                pole_time = millis_to_str(quali["q3"])
            pole_str = pole_name + " (" + pole_time + ")"
        else:
            pole_str = pole_name

        # Fastest lap
        fastest = circuit_fastest_lap_data[circuit_fastest_lap_data["raceId"] == rid]
        fastest = fastest[fastest["rank"] == " 1"]
        if fastest.shape[0] > 0:
            fastest = fastest.iloc[0]
            fastest_name = get_driver_name(fastest["driver_id"])
            fastest_time = fastest["fastest_lap_time_str"]
            fastest_str = fastest_name + " (" + fastest_time + ")"
        else:
            fastest_str = ""

        # Winner and num laps
        win = results[results["position"] == 1].iloc[0]
        winner_name = get_driver_name(win["driverId"])
        winner_time = millis_to_str(win["milliseconds"])
        winner_constructor = get_constructor_name(win["constructorId"])
        num_laps = win["laps"]

        # P2 and P3
        p2_name = results[results["position"] == 2]["driverId"]
        p3_name = results[results["position"] == 3]["driverId"]

        if p2_name.shape[0] > 0:
            p2_name = get_driver_name(p2_name.values[0])
        else:
            p2_name = ""
        if p3_name.shape[0] > 0:
            p3_name = get_driver_name(p3_name.values[0])
        else:
            p3_name = ""

        source = source.append({
            "year": year,
            "laps": num_laps,
            "pole": pole_str,
            "fastest_lap": fastest_str,
            "p1": winner_name + " (" + winner_time + "), " + winner_constructor,
            "p2": p2_name,
            "p3": p3_name
        }, ignore_index=True)

    title_div = Div(text="<h2><b>Circuit Results</b></h2>")

    source = source.sort_values(by="year", ascending=False)
    column_source = ColumnDataSource(data=source)
    results_columns = [
        TableColumn(field="year", title="Year", width=50),
        TableColumn(field="laps", title="Laps", width=50),
        TableColumn(field="pole", title="Pole Position", width=200),
        TableColumn(field="p1", title="First", width=300),
        TableColumn(field="p2", title="Second", width=150),
        TableColumn(field="p3", title="Third", width=150),
    ]
    if source["fastest_lap"].isna().sum() + source[source["fastest_lap"].str.match("")].shape[0] < source.shape[0]:
        results_columns.insert(3, TableColumn(field="fastest_lap", title="Fastest Lap", width=200))
    fast_table = DataTable(source=column_source, columns=results_columns, index_position=None, min_height=530)
    fast_row = row([fast_table], sizing_mode="stretch_width")

    return column([title_div, fast_row], sizing_mode="stretch_width")


def generate_stats_layout(circuit_id, circuit_years, circuit_fastest_lap_data, circuit_results, circuit_races,
                          download_image=True):
    """
    Generates a layout of the circuit image along with some basic stats on the track
    :param circuit_id: Circuit ID
    :param circuit_years: Circuit years
    :param circuit_fastest_lap_data: Circuit fastest lap data
    :param circuit_results: Circuit results
    :param circuit_races: Circuit races
    :param download_image: Whether to download the image
    :return: Race stats layout
    """
    logging.info("Generating circuit stats layout")

    circuit_row = circuits.loc[circuit_id]

    # Track image
    if download_image:
        image_url = str(circuit_row["imgUrl"])
        image_view = plot_image_url(image_url)
        disclaimer = Div(text="The image is of the current configuration of the track.")
        image_view = column([image_view, disclaimer], sizing_mode="stretch_both")
    else:
        image_view = Div()

    # Circuit stats
    header_template = """
    <h2 style="text-align: center;"><b>{}</b></h2>
    """

    template = """
    <pre><b>{}</b> {}<br></pre>
    """

    circuit_stats = header_template.format("Circuit Stats")

    location = circuit_row["location"] + ", " + circuit_row["country"]
    circuit_stats += template.format("Location: ".ljust(22), location)

    years_races = rounds_to_str(circuit_years)
    if len(years_races) > 50:
        years_races = years_races[:50] + "<br>" + "".ljust(22) + years_races[50:]
    circuit_stats += template.format("Years Raced: ".ljust(22), years_races)

    num_races = circuit_races.shape[0]
    circuit_stats += template.format("Races Held: ".ljust(22), str(num_races))

    avg_rating = circuit_row["avgRating"]
    if not np.isnan(avg_rating):
        circuit_stats += template.format("Avg. Fan Rating: ".ljust(22), round(avg_rating, 1))

    # Fastest race lap
    if circuit_fastest_lap_data["fastest_lap_time_millis"].shape[0] > 0 and \
            circuit_fastest_lap_data["fastest_lap_time_millis"].isna().sum() < \
            circuit_fastest_lap_data["fastest_lap_time_millis"].shape[0]:
        minidx = circuit_fastest_lap_data["fastest_lap_time_millis"].idxmin()
        fastest = circuit_fastest_lap_data.loc[minidx]
        did = fastest["driver_id"]
        rid = fastest["raceId"]
        year = str(circuit_races.loc[rid, "year"])
        cid = circuit_results[(circuit_results["raceId"] == rid) &
                              (circuit_results["driverId"] == did)]["constructorId"].values[0]
        constructor = get_constructor_name(cid, include_flag=False)
        fastest = fastest["fastest_lap_time_str"] + " (" + get_driver_name(did) + ", " + constructor + ", " + year + ")"
        circuit_stats += template.format("Fastest Race Lap: ".ljust(22), fastest)

    # DNF pct
    if circuit_results.shape[0] > 0:
        classifications = circuit_results["statusId"].apply(get_status_classification)
        dnfs = classifications[(classifications == "mechanical") | (classifications == "crash")].shape[0]
        finishes = classifications[classifications == "finished"].shape[0]
        dnf_pct = dnfs / (dnfs + finishes)
        dnf_pct_str = f"{round(100 * dnf_pct, 1)}% of cars DNF'd"
        circuit_stats += template.format("DNF Percent: ".ljust(22), dnf_pct_str)

    # Weather and SC Laps
    weather = circuit_races["weather"].value_counts()
    if weather.shape[0] > 0:
        dry = weather["dry"] if "dry" in weather.index else 0
        varied = weather["varied"] if "varied" in weather.index else 0
        wet = weather["wet"] if "wet" in weather.index else 0
        circuit_stats += "<i>Note, we only have safety car and weather data from 2007 to 2017</i>"
        circuit_stats += template.format("Dry Races: ".ljust(22), str(dry) + " (" +
                                         str(round(100 * dry / num_races, 1)) + "% of races)")
        circuit_stats += template.format("Varied Races: ".ljust(22), str(varied) + " (" +
                                         str(round(100 * varied / num_races, 1)) + "% of races)")
        circuit_stats += template.format("Wet Races: ".ljust(22), str(wet) + " (" +
                                         str(round(100 * wet / num_races, 1)) + "% of races)")

    sc_laps = circuit_races["SCLaps"].mean()
    if not np.isnan(sc_laps):
        circuit_stats += template.format("Avg. Safety Car Laps: ".ljust(22), round(sc_laps, 1))

    stats_div = Div(text=circuit_stats)

    divider = vdivider()

    return row([image_view, divider, stats_div], sizing_mode="stretch_width")


def generate_winners_table(circuit_years, circuit_results, circuit_races):
    """
    Table of drivers who've won the most at this circuit
    :return:
    """
    driver_scores = defaultdict(lambda: [0, []])
    constructor_scores = defaultdict(lambda: [0, []])
    for year in circuit_years:
        race = circuit_races[circuit_races["year"] == year]
        rid = race.index.values[0]
        results = circuit_results[circuit_results["raceId"] == rid]
        win = results[results["position"] == 1]
        if win.shape[0] > 0:
            win = win.iloc[0]
        else:
            continue
        driver_winner_name = get_driver_name(win["driverId"])
        driver_scores[driver_winner_name][0] += 1
        driver_scores[driver_winner_name][1].append(year)
        constructor_winner_name = get_constructor_name(win["constructorId"])
        constructor_scores[constructor_winner_name][0] += 1
        constructor_scores[constructor_winner_name][1].append(year)
    driver_scores = pd.DataFrame.from_dict(driver_scores, orient="index", columns=["wins", "years"])
    constructor_scores = pd.DataFrame.from_dict(constructor_scores, orient="index", columns=["wins", "years"])
    driver_scores.index.name = "name"
    constructor_scores.index.name = "name"
    driver_scores = driver_scores.sort_values(by="wins", ascending=False)
    constructor_scores = constructor_scores.sort_values(by="wins", ascending=False)
    driver_scores["years"] = driver_scores["years"].apply(rounds_to_str)
    constructor_scores["years"] = constructor_scores["years"].apply(rounds_to_str)

    winners_columns = [
        TableColumn(field="wins", title="Num. Wins", width=50),
        TableColumn(field="name", title="Name", width=200),
        TableColumn(field="years", title="Years Won", width=200),
    ]

    # Driver table
    title_div = Div(text=u"<h2><b>Who has won the most at this circuit \u2014 Drivers?</b></h2>")
    source = ColumnDataSource(data=driver_scores)
    driver_winners_table = DataTable(source=source, columns=winners_columns, index_position=None, min_height=530)
    driver_winners_layout = column([title_div, row([driver_winners_table], sizing_mode="stretch_width")])

    # Constructor table
    title_div = Div(text=u"<h2><b>Who has won the most at this circuit \u2014 Constructors?</b></h2>")
    source = ColumnDataSource(data=constructor_scores)
    constructor_winners_table = DataTable(source=source, columns=winners_columns, index_position=None, min_height=530)
    constructor_winners_layout = column([title_div, row([constructor_winners_table], sizing_mode="stretch_width")])

    return row([driver_winners_layout, vdivider(), constructor_winners_layout], sizing_mode="stretch_width")


def generate_error_layout():
    """
    Generates an error layout in the event that the user selects an invalid circuit.
    :return: Error layout
    """
    text = "Somehow, you have selected an invalid circuit. The circuits we have data on are..."
    text += "<ul>"
    for cid in circuits.index:
        name = get_circuit_name(cid, include_flag=False)
        text += f"<li>{name}</li>"
    text += "</ul><br>"
    return Div(text=text)
