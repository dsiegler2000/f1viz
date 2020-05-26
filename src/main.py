from bokeh.layouts import column, row
from bokeh.models import AutocompleteInput, Div, Select
from bokeh.io import curdoc
from data_loading.data_loader import load_races, load_drivers, load_circuits, load_constructors
from mode import home, yearcircuit, unimplemented, year, driver, circuit, constructor, circuitdriver, driverconstructor, \
    yeardriver, yearconstructor, circuitconstructor, yearcircuitdriver, yearcircuitconstructor, yeardriverconstructor
import os
import logging

# Set up some logging
logging.basicConfig(level=logging.NOTSET)
logging.root.setLevel(logging.NOTSET)

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
    0b1100: ["YEARCIRCUIT", yearcircuit],    # a year plus a circuit = a race
    0b0110: ["CIRCUITDRIVER", circuitdriver],
    0b0011: ["DRIVERCONSTRUCTOR", driverconstructor],
    0b1010: ["YEARDRIVER", yeardriver],
    0b1001: ["YEARCONSTRUCTOR", yearconstructor],
    0b0101: ["CIRCUITCONSTRUCTOR", circuitconstructor],
    0b1110: ["YEARCIRCUITDRIVER", yearcircuitdriver],
    0b1101: ["YEARCIRCUITCONSTRUCTOR", yearcircuitconstructor],
    0b1011: ["YEARDRIVERCONSTRUCTOR", yeardriverconstructor],
    0b0111: "CIRCUITDRIVERCONSTRUCTOR",      # doesn't always make sense
    0b1111: "YEARCIRCUITDRIVERCONSTRUCTOR",  # doesn't always make sense
}
mode_lay = unimplemented.get_layout()
mode = "default"


def update():
    global mode_lay, lay, mode

    year_v = year_input.value
    circuit_v = circuit_input.value
    driver_v = driver_input.value
    constructor_v = constructor_input.value

    year_v = "XXXX" if year_v.startswith("<select") else year_v
    circuit_v = "XXXX" if circuit_v.startswith("<select") else circuit_v
    driver_v = "XXXX" if driver_v.startswith("<select") else driver_v
    constructor_v = "XXXX" if constructor_v.startswith("<select") else constructor_v

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

    logging.info(f"Updating to mode: {mode[0]}...")

    mode_lay = mode[1].get_layout(year_id=year_id, circuit_id=circuit_id,
                                  driver_id=driver_id, constructor_id=constructor_id, mode=mode[0])

    curdoc().remove_root(lay)
    lay = column(desc, search_bars_row, mode_lay, sizing_mode="scale_width")
    curdoc().add_root(lay)


logging.info("Constructing initial layout...")

# Header
desc = Div(text=open(os.path.join("src", "header.html")).read(), sizing_mode="stretch_width")

# Season search bar
year_completions = ["<select year>"] + [str(y) for y in races.sort_values(by="year", ascending=False)["year"].unique()]
year_completions.remove("2020")
year_input = Select(options=year_completions)

# Circuit / race
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
racecircuit_completions = ["<select race or circuit>"] + race_completions + circuit_completions
circuit_input = Select(options=racecircuit_completions)

# Driver
driver_completions = ["<select driver>"]
for i, r in drivers.iterrows():
    driver_completions.append(r["forename"] + " " + r["surname"])
driver_input = Select(options=driver_completions)

# Constructor
constructor_completions = ["<select constructor>"] + [c for c in constructors["name"].unique()]
constructor_input = Select(options=constructor_completions)


driver_input.value = "Lewis Hamilton"
year_input.value = "2017"

search_bars = [circuit_input, year_input, driver_input, constructor_input]
for s in search_bars:
    s.on_change("value", lambda attr, old, new: update())

search_bars_row = row(*search_bars, sizing_mode="scale_width")

lay = column(desc, search_bars_row, mode_lay, sizing_mode="scale_width")

curdoc().add_root(lay)
curdoc().title = "F1Viz"
curdoc().theme = "dark_minimal"

constructor_input.value = "Mercedes"

logging.info("Initialized")
