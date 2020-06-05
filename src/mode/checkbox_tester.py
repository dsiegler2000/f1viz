import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from utils import generate_plot_list_selector, PlotItem


def get_layout(**kwargs):
    group = generate_plot_list_selector([PlotItem(generate_sin_plot, [], {}, "Sin plot :)")])
    return group


def generate_sin_plot():
    x = np.linspace(0, 4 * np.pi, 200)
    y = np.sin(x)
    source = ColumnDataSource(data=dict(x=x, y=y))

    # Set up plot
    plot = figure(plot_height=400, plot_width=400, title="my sine wave",
                  tools="crosshair,pan,reset,save,wheel_zoom",
                  x_range=[0, 4 * np.pi], y_range=[-2.5, 2.5])

    plot.line("x", "y", source=source, line_width=3, line_alpha=0.6)
    return plot
