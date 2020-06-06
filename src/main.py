from bokeh.layouts import column, row
from bokeh.models import AutocompleteInput, Div, Select
from bokeh.io import curdoc
from data_loading.data_loader import load_races, load_drivers, load_circuits, load_constructors
from mode import home, yearcircuit, unimplemented, year, driver, circuit, constructor, circuitdriver, \
    driverconstructor, yeardriver, yearconstructor, circuitconstructor, yearcircuitdriver, yearcircuitconstructor, \
    yeardriverconstructor, circuitdriverconstructor, yearcircuitdriverconstructor, allyears
import os
import logging

# Set up some logging
logging.basicConfig(level=logging.NOTSET)
logging.root.setLevel(logging.NOTSET)

logging.info(f"Receiving request, PID: {os.getpid()}")

# Load data
races = load_races()
circuits = load_circuits()
drivers = load_drivers()
constructors = load_constructors()

logging.info("Loaded data")

modes = {
    # year, circuit, driver, constructor
    0b0000: ["HOME", home],
    0b1000: ["YEAR", year],
    0b0100: ["CIRCUIT", circuit],
    0b0010: ["DRIVER", driver],
    0b0001: ["CONSTRUCTOR", constructor],
    0b1100: ["YEARCIRCUIT", yearcircuit],
    0b0110: ["CIRCUITDRIVER", circuitdriver],
    0b0011: ["DRIVERCONSTRUCTOR", driverconstructor],
    0b1010: ["YEARDRIVER", yeardriver],
    0b1001: ["YEARCONSTRUCTOR", yearconstructor],
    0b0101: ["CIRCUITCONSTRUCTOR", circuitconstructor],
    0b1110: ["YEARCIRCUITDRIVER", yearcircuitdriver],
    0b1101: ["YEARCIRCUITCONSTRUCTOR", yearcircuitconstructor],
    0b1011: ["YEARDRIVERCONSTRUCTOR", yeardriverconstructor],
    0b0111: ["CIRCUITDRIVERCONSTRUCTOR", circuitdriverconstructor],
    0b1111: ["YEARCIRCUITDRIVERCONSTRUCTOR", yearcircuitdriverconstructor],
    "all_years": ["ALLYEARS", allyears],
    "all_circuits": None,
    "all_drivers": None,
    "all_constructors": None
}

year_completions = ["<select year>", "All Years"] + [str(y) for y in
                                                     races.sort_values(by="year", ascending=False)["year"].unique()]

race_completions = []
for i, r in races.iterrows():
    race_name = r["name"]
    races_with_same_name = races[races["name"].str.lower() == race_name.lower()]
    circuit_ids = races_with_same_name["circuitId"]
    num_circuits = len(list(circuit_ids.unique()))
    to_add = []
    if num_circuits == 1:
        to_add.append(race_name)
    else:
        for j, value in circuit_ids.value_counts().sort_values(ascending=False).iteritems():
            circuit_name = circuits.loc[j, "name"]
            to_add.append(race_name + " at " + circuit_name)
    for n in to_add:
        if n not in race_completions:
            race_completions.append(n)
circuit_completions = [c for c in circuits["name"].unique()]
racecircuit_completions = ["<select race or circuit>", "All Circuits"] + race_completions + circuit_completions

driver_completions = ["<select driver>", "All Drivers"]
for i, r in drivers.iterrows():
    driver_completions.append(r["forename"] + " " + r["surname"])

constructor_completions = ["<select constructor>", "All Constructors"] + [c for c in constructors["name"].unique()]


# TODO master list:
#  Go through each existing mode and do a "second pass" to add simple features and make small clean-up changes      √
#   Make sure tables are sortable                                                                                   √
#   Make sure second axes are scaled properly                                                                       √
#   Make sure using ordinals (1st, 2nd, 3rd) on everything                                                          √
#   Make sure the mode has a header                                                                                 √
#  Get rid of axis sharing on whatever mode that is                                                                 √
#  Add axis overrides to position plot, SP v FP, MLTR vs FP, and any other plots to make the axes ordinal           √
#   Try this out on one mode, see how it feels then make a decision                                                 √
#  Change all mean lap time ranks to be mean lap time percent (except in position plot)
#  Add the top-n support for all win plots as well as the calculate 95th percentile and set that as n feature
#  Add smoothing slider to positions plots
#  Check all stats divs for things that need to be `strip`-ed
#  Add the plots checklists for efficiency, make it into a class so it's easy to implement, see Trello and utils
#   home                                    √
#   year                                    √
#   circuit                                 √
#   driver                                  √
#   constructor                             √
#   yearcircuit
#   circuitdriver
#   driverconstructor
#   yeardriver
#   yearconstructor
#   circuitconstructor
#   yearcircuitdriver
#   yearcircuitconstructor
#   yeardriverconstructor
#   circuitdriverconstructor
#   yearcircuitdriverconstructor
#  Release to r/Formula1 (without the all_ modes)
#  Start on the all_years or home mode


def _get_mode(year_input, circuit_input, driver_input, constructor_input):
    year_v = year_input.value
    circuit_v = circuit_input.value
    driver_v = driver_input.value
    constructor_v = constructor_input.value

    year_v = "XXXX" if year_v.startswith("<select") else year_v
    circuit_v = "XXXX" if circuit_v.startswith("<select") else circuit_v
    driver_v = "XXXX" if driver_v.startswith("<select") else driver_v
    constructor_v = "XXXX" if constructor_v.startswith("<select") else constructor_v

    all_years = year_v.lower() == "all years"
    all_circuits = circuit_v.lower() == "all circuits"
    all_drivers = driver_v.lower() == "all drivers"
    all_constructors = constructor_v.lower() == "all constructors"

    year_id = -1
    circuit_id = -1
    driver_id = -1
    constructor_id = -1

    if all_years:
        mode = modes["all_years"]
    elif all_circuits:
        mode = modes["all_circuits"]
    elif all_drivers:
        mode = modes["all_drivers"]
    elif all_constructors:
        mode = modes["all_constructors"]
    else:
        year_in = year_v in year_completions
        circuit_in = circuit_v in racecircuit_completions
        driver_in = driver_v in driver_completions
        constructor_in = constructor_v in constructor_completions
        mode = modes[(year_in << 3) + (circuit_in << 2) + (driver_in << 1) + (constructor_in << 0)]

        # Determine the year id (just the year)
        year_id = int(year_v) if year_in else -1

        # Determine the circuit id
        circuit_id = -1
        if circuit_in:
            if circuit_v in race_completions:
                split = circuit_v.split(" at ")
                race_name = split[0]
                if len(split) == 1:
                    circuit_id = races[races["name"].str.lower() == race_name.lower()]["circuitId"].unique()[0]
                else:
                    circuit_name = split[1]
                    circuit_id = circuits[circuits["name"].str.lower() == circuit_name.lower()].index.unique()[0]
            elif circuit_v in circuit_completions:
                circuit_id = circuits[circuits["name"].str.lower() == circuit_v.lower()].index.unique()[0]
        circuit_id = int(circuit_id)

        # Determine the driver id
        split = driver_v.split(" ")
        driver_id = -1
        if driver_in:
            for first_name_len in range(1, len(split)):
                first_name = " ".join(split[:first_name_len])
                last_name = " ".join(split[first_name_len:])
                matched = drivers[(drivers["forename"].str.lower() == first_name.lower()) &
                                  (drivers["surname"].str.lower() == last_name.lower())]
                if matched.shape[0] == 1:
                    driver_id = matched.index.values[0]
                    break
        driver_id = int(driver_id)

        # Determine the constructor id
        constructor_id = int(constructors[constructors["name"].str.lower() == constructor_v.lower()].index.unique()[0])\
            if constructor_in else -1

    # Dispatch to the proper module
    if not isinstance(mode, list):
        mode = ["unimplemented", unimplemented]

    return mode, year_id, circuit_id, driver_id, constructor_id


def _update(year_input, circuit_input, driver_input, constructor_input):
    mode, year_id, circuit_id, driver_id, constructor_id = _get_mode(year_input, circuit_input, driver_input,
                                                                     constructor_input)
    logging.info(f"Updating to mode: {mode[0]}...")
    plots_layout = mode[1].get_layout(year_id=year_id, circuit_id=circuit_id, driver_id=driver_id,
                                      constructor_id=constructor_id, mode=mode[0])
    generate_main(plots_layout, year_v=year_input.value, circuit_v=circuit_input.value, driver_v=driver_input.value,
                  constructor_v=constructor_input.value)


def generate_main(plots_layout, year_v=None, circuit_v=None, driver_v=None, constructor_v=None, first_time=False):
    logging.info(f"Generating main, year_v={year_v}, circuit_v={circuit_v}, driver_v={driver_v}, "
                 f"constructor_v={constructor_v}")
    # Header and footer
    header = Div(text=open(os.path.join("src", "header.html")).read(), sizing_mode="stretch_width")
    footer = Div(text=open(os.path.join("src", "footer.html")).read(), sizing_mode="stretch_width")

    year_input = Select(options=year_completions)
    circuit_input = Select(options=racecircuit_completions)
    driver_input = Select(options=driver_completions)
    constructor_input = Select(options=constructor_completions)

    if year_v:
        year_input.value = year_v
    if circuit_v:
        circuit_input.value = circuit_v
    if driver_v:
        driver_input.value = driver_v
    if constructor_v:
        constructor_input.value = constructor_v

    search_bars = [circuit_input, year_input, driver_input, constructor_input]
    search_bars_layout = row(*search_bars, sizing_mode="scale_width")

    search_bars_layout = column([search_bars_layout], sizing_mode="scale_width")

    lay = column([header, search_bars_layout, plots_layout, footer], sizing_mode="scale_width")

    curdoc().clear()
    curdoc().add_root(lay)
    curdoc().title = "F1Viz"
    curdoc().theme = "dark_minimal"

    for s in search_bars:
        s.on_change("value", lambda attr, old, new: _update(year_input, circuit_input, driver_input, constructor_input))

    if first_time:
        _update(year_input, circuit_input, driver_input, constructor_input)
        # Put any default values here


logging.info("Constructing initial layout...")
generate_main(Div(), first_time=True)
logging.info("Initialized")
