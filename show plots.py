import pandas as pd

from Model import *
from Classes import track_statistics
from Parameters import parameter


from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.layouts import row


def plot_cov(tracker, timesteps):
    source = ColumnDataSource(data={'time': [t for t in range(timesteps)],
                                    'infected': tracker['currently infected'],
                                    'deceased': tracker['deceased'],
                                    'recovered': tracker['recovered'],
                                    'transmitter': tracker['transmitter'],
                                    'symptomatic': tracker['symptomatic'],
                                    'hospitalized': tracker['hospitalized'],
                                    'vaccinated': tracker['vaccinated'],
                                    'total_infected': tracker['total infected']
                                    }
                              )
    p1 = figure(
        title="Title",
        x_axis_label='days',
        y_axis_label='people',
        tools='reset,save,pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out')

    # add a line renderer with source
    p1.line(
        x='time',
        y='infected',
        legend_label='infected',
        line_width=1,
        line_color="red",
        source=source)

    p1.line(
        x='time',
        y='hospitalized',
        legend_label='hospitalized',
        line_width=1,
        line_color="purple",
        source=source)

    p1.line(
        x='time',
        y='transmitter',
        legend_label='transmitter',
        line_width=1,
        line_color="orange",
        source=source)

    p1.line(
        x='time',
        y='symptomatic',
        legend_label='symptomatic',
        line_width=1,
        line_color="green",
        source=source)

    p1.add_tools(
        HoverTool(
            tooltips=[('time', '@time'),
                      ('infected', '@infected'),
                      ('transmitter', '@transmitter'),
                      ('symptomatic', '@symptomatic'),
                      ('hospitalized', '@hospitalized'),
                      ('vaccinated', '@vaccinated')]))

    p1.legend.orientation = "horizontal"

    p2 = figure(
        title="recovered",
        x_axis_label='days',
        y_axis_label='people',
        tools='reset,save,pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out')

    p2.line(
        x='time',
        y='total_infected',
        legend_label='total infected',
        line_width=1,
        line_color="red",
        source=source)

    p2.line(
        x='time',
        y='recovered',
        legend_label='recovered',
        line_width=1,
        line_color="green",
        source=source)

    p2.line(
        x='time',
        y='vaccinated',
        legend_label='vaccinated',
        line_width=1,
        line_color="blue",
        source=source)

    p2.line(
        x='time',
        y='deceased',
        legend_label='deceased',
        line_width=1,
        line_color="orange",
        source=source)

    p2.add_tools(
        HoverTool(
            tooltips=[('time', '@time'),
                      ('recovered', '@recovered'),
                      ('total infected', '@total_infected'),
                      ('deceased', '@deceased')]))

    p2.legend.orientation = "horizontal"
    # show the results
    show(row(p1, p2))


data = pd.read_csv("Custom_total.csv")
plot_cov(data, 300)
#data = pd.read_csv("Old_Young_total.csv")
#plot_cov(data, 300)
#data = pd.read_csv("Danish_total.csv")
#plot_cov(data, 300)

