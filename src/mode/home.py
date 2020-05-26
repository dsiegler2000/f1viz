import os
from bokeh.models import Div


def get_layout(**kwargs):
    html = """
    <p><i><b>
        Fill out any combination of race or circuit, year, driver, and/or constructor to see some data!
    </b></i></p>
    """
    desc = Div(text=html)
    return desc
