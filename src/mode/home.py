from bokeh.models import Div


def get_layout(**kwargs):
    html = """
    <p><i><b>
        Fill out any combination of race or circuit, year, driver, and/or constructor, select which visualizations you 
        want to see, then click "Generate Plots" to start (then wait for a few seconds...)
    </b></i></p>
    """
    desc = Div(text=html)
    return desc
