import itertools
import logging
import time
from collections import defaultdict
from io import BytesIO
import numpy as np
import flag
import requests
from PIL import Image
from bokeh.colors import RGB
from bokeh.palettes import Set3_12 as palette
from bokeh.layouts import row, column
from bokeh.models import Div, Spacer, DatetimeTickFormatter
from bokeh.plotting import figure
import pandas as pd
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg
from data_loading.data_loader import load_drivers, load_constructors, load_circuits, load_races, \
    load_constructor_colors, load_status

# This is the color of the Bokeh plots
PLOT_BACKGROUND_COLOR = RGB(22, 25, 28)
BASE_BACKGROUND_COLOR = RGB(47, 47, 47)

LINK_TEMPLATE = """<a href="{}">{}</a>"""

_MINSEC_FORMAT = "%M:%S.%3N"
DATETIME_TICK_KWARGS = dict(
    minsec=_MINSEC_FORMAT,
    milliseconds=_MINSEC_FORMAT,
    seconds=_MINSEC_FORMAT,
    minutes=_MINSEC_FORMAT,
    hourmin=_MINSEC_FORMAT
)

drivers = load_drivers()
constructors = load_constructors()
circuits = load_circuits()
races = load_races()
constructor_colors = load_constructor_colors()
status = load_status()

# Quick map from nationality to their country code (for flag emojis)
NATIONALITY_TO_FLAG = pd.read_csv("data/static_data/nationalities.csv", index_col=0)


def plot_image_url(image_url):
    """
    Generates a plot of the image found at the given URL, removing all styling from the plot.
    :param image_url:
    :return:
    """
    if image_url == "nan":
        return Div(text="No image found.")

    try:
        # Get the image
        if image_url.endswith(".svg"):
            img = svg_to_pil(image_url)
        else:
            img = Image.open(BytesIO(requests.get(image_url).content))

        w, h = img.size
        img = img.convert("RGBA")
        img = np.array(img)
        img = np.flipud(img)

        # First attempt and constrast/blacks adjusting
        # mask = np.zeros(img.shape)
        # cv2.inRange(img, 0, 25, mask)
        # img = cv2.bitwise_and(img, cv2.bitwise_not(mask))
        # mask = 255 - mask
        # img += mask

        image_view = figure(aspect_ratio=w / h, tools="")
        image_view.image_rgba(image=[img], x=0, y=0, dw=1, dh=1)

        # Remove all styling
        image_view.xgrid.grid_line_color = None
        image_view.ygrid.grid_line_color = None
        image_view.toolbar.logo = None
        image_view.toolbar_location = None
        image_view.axis.visible = False
        image_view.background_fill_alpha = 0

        image_view.border_fill_alpha = 0
        image_view.outline_line_alpha = 0

        return image_view
    except:
        return Div(text="Error retrieving image.")


def svg_to_pil(url):
    """
    SVG URL to PIL image
    :param url: URL
    :return: PIL image
    """
    logger = logging.getLogger("svglib")
    logger.disabled = True
    out = BytesIO()
    drawing = svg2rlg(BytesIO(requests.get(url).content))
    renderPM.drawToFile(drawing, out, fmt="PNG", bg=0x2F2F2F)
    image = Image.open(out)
    return image


def millis_to_str(millis, format_seconds=False, fallback=""):
    """
    Converts millisecond value to a timestamp string, returning `fallback` if the conversion failed.
    :param millis: Milliseconds
    :param format_seconds: If set to true and millis < 1000 * 60, the returned string will just be the seconds
    :param fallback: Fallback string to return
    :return: Converted string
    """
    dt = pd.to_datetime(millis, unit="ms", errors="ignore")
    if pd.isnull(dt):
        return fallback
    else:
        if millis < 1000 * 60 and format_seconds:
            return dt.strftime("%S.%f")[:-3]
        if millis > 1000 * 60 * 60:
            return dt.strftime("%-H:%M:%S.%f")[:-3]
        else:
            return dt.strftime("%M:%S.%f")[:-3]


def get_driver_name(did, include_flag=True, just_last=False):
    """
    Gets the stylized version of the given driver's name
    :param did: Driver ID
    :param include_flag: Whether to include the nationality flag in the driver's name
    :param just_last: Whether to include the first name
    :return: String
    """
    driver = drivers.loc[did]
    if just_last:
        name = driver["surname"]
    else:
        name = driver["forename"] + " " + driver["surname"]
    if include_flag:
        nat = driver["nationality"].lower()
        if nat in NATIONALITY_TO_FLAG.index:
            flag_t = NATIONALITY_TO_FLAG.loc[nat, "flag"]
            name = flag.flagize(f":{flag_t}: " + name)
        else:
            logging.warning(f"Unknown nationality {nat}, driver ID: {did}")
    return name


def nationality_to_flag(nat):
    flag_t = NATIONALITY_TO_FLAG.loc[nat, "flag"]
    return flag.flagize(f":{flag_t}:")


def get_constructor_name(cid, include_flag=True):
    """
    Gets the stylized version of the given constructor's name
    :param cid: Constructor ID
    :param include_flag: Whether to include the nationality flag in the constructor's name
    :return: String
    """
    try:
        constructor = constructors.loc[cid]
        name = constructor["name"]
        if include_flag:
            nat = constructor["nationality"].lower()
            if nat in NATIONALITY_TO_FLAG.index:
                flag_t = NATIONALITY_TO_FLAG.loc[nat, "flag"]
                name = flag.flagize(f":{flag_t}: " + name)
            else:
                logging.warning(f"Unknown nationality {nat}, constructor ID: {cid}")
        return name
    except KeyError:
        return "UNKNOWN"


def get_circuit_name(cid, include_flag=True):
    """
    Gets the stylized version of the given circuit's name
    :param cid: Circuit ID
    :param include_flag: Whether to include the nationality flag in the constructor's name
    :return: String
    """
    circuit = circuits.loc[cid]
    name = circuit["name"]
    if include_flag:
        nat = circuit["country"].lower()
        if nat in NATIONALITY_TO_FLAG.index:
            flag_t = NATIONALITY_TO_FLAG.loc[nat, "flag"]
            name = flag.flagize(f":{flag_t}: " + name)
        else:
            logging.warning(f"Unknown nationality {nat}, circuit ID: {cid}")
    return name


def get_race_name(rid, include_flag=True, include_country=True, line_br=None, use_shortened=False, include_year=False):
    """
    Gets the stylized version of the given race's name
    :param rid: Race ID
    :param include_flag: Whether to include the nationality flag in the constructor's name
    :param include_country: Whether to use the full race name or just the country name
    :param line_br: Character to use for line break, or None if not desired
    :param use_shortened: Use shortened version of GP name
    :param include_year: Whether to include the year in the name, only works if `use_shortened` and `line_br` are both
    False
    :return: String
    """
    race = races.loc[rid]
    circuit = circuits.loc[race["circuitId"]]
    name = circuit["country"] if include_country else race["name"]
    if use_shortened:
        name = name[:3].upper()
    if include_flag:
        nat = circuit["country"].lower()
        if nat in NATIONALITY_TO_FLAG.index:
            flag_t = NATIONALITY_TO_FLAG.loc[nat, "flag"]
            if line_br:
                name = flag.flagize(f"{name} {line_br} :{flag_t}:")
            else:
                name = flag.flagize(f":{flag_t}: " + name)
        else:
            logging.warning(f"Unknown nationality {nat}, race ID: {rid}")
    if include_year:
        name = str(race["year"]) + " " + name
    return name


def vdivider(line_thickness=1, border_thickness=3, line_color="white", border_color=BASE_BACKGROUND_COLOR):
    """
    Generates a vertical divider (spacer with a line)
    :param line_thickness: Thickness of the line
    :param border_thickness: Thickness of the border, total thickness = `border_thickness * 2 + line_thickness`
    :param line_color: Line color
    :param border_color: Border color
    :return: The layout
    """
    divider = Spacer(width=border_thickness, background=border_color)
    divider = row([divider, Spacer(width=line_thickness, background=line_color), divider], sizing_mode="stretch_height")
    return divider


def hdivider(line_thickness=2, border_thickness=6, top_border_thickness=6, bottom_border_thickness=6,
             line_color="white", border_color=BASE_BACKGROUND_COLOR):
    """
    Generates a horizontal divider (spacer with a line)
    :param line_thickness: Thickness of the line
    :param border_thickness: Thickness of the border, total thickness = `border_thickness * 2 + line_thickness`,
    overrides top and bottom border thickness
    :param top_border_thickness: Thickness of the top border
    :param bottom_border_thickness: Thickness of the bottom border
    :param line_color: Line color
    :param border_color: Border color
    :return: The layout
    """
    if border_thickness != 6:
        top = border_thickness
        bottom = border_thickness
    else:
        top = top_border_thickness
        bottom = bottom_border_thickness
    divider_top = Spacer(height=top, background=border_color)
    divider_bottom = Spacer(height=bottom, background=border_color)
    divider = column([divider_top, Spacer(height=line_thickness, background=line_color), divider_bottom],
                     sizing_mode="stretch_width")
    return divider


def linkify(url, text=None):
    """
    Converts a URL to an <a> tag
    :param url: URL
    :param text: Text for the link, or None if using the URL
    :return: Link
    """
    return LINK_TEMPLATE.format(url, text if text else url)


def get_line_thickness(position, min_thickness=1.8, max_thickness=2.7):
    """
    Gets line thickness for the given position
    :param position: Position
    :param min_thickness: Min thickness
    :param max_thickness: Max thickness
    :return: Thickness
    """
    if position == "DNF":
        return min_thickness
    # Simply lerp
    m = (max_thickness - min_thickness) / 20
    return max(min_thickness, min(max_thickness, -m * int(position) + max_thickness))


def time_decorator(f):
    def timed(*args, **kwargs):
        start = time.time()
        to_return = f(*args, **kwargs)
        end = time.time()
        elapsed = 1000 * (end - start)
        print(f"Completed {f.__name__} in  {elapsed} milliseconds")
        return to_return
    return timed


def str_to_millis(s):
    """
    Quick and dirty method to convert the timestamp to milliseconds.
    """
    if s == r"\N":
        return np.nan
    if not isinstance(s, str):
        return s
    s = s.strip().replace("+", "").replace("sec", "").replace("s", "")
    if ":" in s:  # 01:32:329
        split = s.split(":")
        millis = int(split[0]) * 60 * 1000
        if "." in split[1]:
            split = split[1].split(".")
        elif len(split) == 2:
            split = [split[1], "0"]
        else:
            split = [split[1], split[2]]
        millis += int(split[0]) * 1000
        millis += int(split[1].ljust(3, "0"))
        return millis
    elif "." in s:  # 5.293
        return int(float(s) * 1000)
    else:
        return None


def position_text_to_str(pos):
    pos = pos.lower()
    if pos == "r":
        pos = "RET"
    if pos == "f":
        pos = "DNQ"
    if pos == "n":
        pos = "NC"
    if pos == "d":
        pos = "DSQ"
    if pos == "e":
        pos = "DSQ"
    if pos == "w":
        pos = "DNQ"
    return pos


def get_status_classification(status):
    if status in [1, 2, 11, 12, 13, 14, 15, 16, 17, 18, 19, 45, 50, 128, 53, 55, 58, 88, 111, 112, 113, 114, 115,
                  116, 117, 118, 119, 120, 122, 123, 124, 125, 127, 133, 134]:
        return "finished"
    elif status in [3, 4, 20]:
        return "crash"
    elif status in [81, 97, 54, 77]:
        return "dnq"
    else:
        return "mechanical"


def rescale(s, new_min, new_max):
    new_range = new_max - new_min
    old_range = s.max() - s.min()
    return new_min + new_range * (s - s.min()) / old_range


def rounds_to_str(rounds, all_shape=1000):
    rounds = sorted(rounds)
    # Some base cases
    if len(rounds) == 0:
        return "None"
    if len(rounds) == all_shape:
        return "All"
    if len(rounds) == 1:
        return str(rounds[0])
    rounds = sorted(rounds)
    runs = [[rounds[0], rounds[0]]]  # [start, end]
    j = 0
    for i, round in enumerate(rounds):
        if round == runs[j][1] + 1:  # Extend the run
            runs[j][1] = round
        else:  # New run
            runs.append([round, round])
            j += 1
    runs = runs[1:]
    out = ""
    for run in runs:
        if run[0] == run[1]:
            out += str(run[0]) + ", "
        else:
            out += str(run[0]) + "-" + str(run[1]) + ", "
    return out[:-2]


def int_to_ordinal(n):
    if n is None:
        return ""
    if isinstance(n, str) and n.strip().isnumeric():
        n = int(n)
    if np.isnan(n):
        return ""
    else:
        return "%d%s" % (n, {1: "st", 2: "nd", 3: "rd"}.get(n if n < 20 else n % 10, "th"))


def result_to_str(pos, status_id):
    """
    Converts a position and a status into a results string by converting position to ordinal or marking as RET with
    the reason if status indicates that the driver retired.
    :param pos: Position (official classification)
    :param status_id: Status ID
    :return: Results string, finish position (if driver retired, will be nan)
    """
    finish_pos = str(pos)
    classification = get_status_classification(status_id)
    if classification == "finished" and finish_pos.isnumeric():
        finish_pos = int(finish_pos)
        finish_pos_str = int_to_ordinal(finish_pos)
    else:
        finish_pos = np.nan
        finish_pos_str = "RET (" + status.loc[status_id, "status"] + ")"
    return finish_pos_str, finish_pos


class ColorDashGenerator:
    """
    This class does NOT follow the regular generator template.
    """
    def __init__(self, colors=palette, dashes=None):
        if dashes is None:
            dashes = ["solid", "dashed", "dotted", "dotdash"]
        self.constructor_color_map = {}
        self.constructor_dashes_map = defaultdict(lambda: itertools.cycle(dashes))
        self.driver_dashes_map = {}
        self.colors = itertools.cycle(colors)

    def get_color_dash(self, did, cid):
        # First, check if we have the constructor in the csv
        if cid in constructor_colors.index.values:
            color = constructor_colors.loc[cid]
            color = RGB(color["R"], color["G"], color["B"])
        else:
            if cid in self.constructor_color_map.keys():
                color = self.constructor_color_map[cid]
            else:
                color = self.colors.__next__()
                self.constructor_color_map[cid] = color

        # Now get the line dash
        if did is None:
            dash = "solid"
        elif did in self.driver_dashes_map.keys():
            dash = self.driver_dashes_map[did]
        else:
            dash = self.constructor_dashes_map[cid].__next__()
            self.driver_dashes_map[did] = dash

        return color, dash

