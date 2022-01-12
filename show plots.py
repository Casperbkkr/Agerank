import pandas as pd
import os
from Model import *
from Classes import track_statistics
from Parameters import parameter

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.layouts import row



def plot_one_strategy(tracker: pd.DataFrame, timesteps: int) -> None:
    # function for plotting a single results file.
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

    p1.legend.orientation = "vertical"

    p2 = figure(
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

    p2.legend.orientation = "vertical"
    p2.legend.location = "top_left"
    # show the results
    show(row(p1, p2))


# function for plotting five strategies concurrently.
def plot_all_strategies(trackerYO: pd.DataFrame, trackerOY: pd.DataFrame,
                        tracker10: pd.DataFrame, tracker30: pd.DataFrame,
                        tracker50: pd.DataFrame, timesteps: int, to_show: str) \
                        -> None:

    # function for plotting five strategies concurrently.
    # takes pandas dataframes of same size as input and the property that needs to be show
    source = ColumnDataSource(data={'time': [t for t in range(timesteps)],
                                    'total_infected_YO': trackerYO[to_show],
                                    'total_infected_OY': trackerOY[to_show],
                                    'total_infected_10': tracker10[to_show],
                                    'total_infected_30': tracker30[to_show],
                                    'total_infected_50': tracker50[to_show]
                                    }
                              )

    p2 = figure(
        x_axis_label='days',
        y_axis_label='people',
        tools='reset,save,pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out')

    p2.line(
        x='time',
        y='total_infected_YO',
        legend_label='Young old',
        line_width=1,
        line_color="red",
        source=source)

    p2.line(
        x='time',
        y='total_infected_OY',
        legend_label='Old young',
        line_width=1,
        line_color="black",
        source=source)

    p2.line(
        x='time',
        y='total_infected_10',
        legend_label='Mix-10',
        line_width=1,
        line_color="blue",
        source=source)

    p2.line(
        x='time',
        y='total_infected_30',
        legend_label='Mix-30',
        line_width=1,
        line_color="green",
        source=source)

    p2.line(
        x='time',
        y='total_infected_50',
        legend_label='Mix-50',
        line_width=1,
        line_color="purple",
        source=source)

    p2.legend.orientation = "vertical"
    p2.legend.location = "bottom_right"
    # show the results
    show(p2)


def read_results(filename1: str) -> pd.DataFrame:
    # function for reading results.
    # input is a filename without path
    # returns a pandas dataframe
    filename2 = filename1 + ".csv"
    filename3 = os.path.join("Results", filename2)
    return pd.read_csv(filename3)


# What file to show and how many timesteps. Default results are 400 timesteps max.
timesteps = 400
filename = "Young_old_uniform_18"
# show results
results = read_results(filename)
plot_one_strategy(results, timesteps)

# Change names of results to be shown concurrently
filename1 = "Young_old_total"
filename2 = "Old_young_uniform_total"
filename3 = "ISR-10_uniform_total"
filename4 = "ISR-30_uniform_total"
filename5 = "ISR-50_uniform_total"

# Read the results
results1 = read_results(filename1)
results2 = read_results(filename2)
results3 = read_results(filename3)
results4 = read_results(filename4)
results5 = read_results(filename5)

# What property do you want to see?
# Choose from: "susceptible", "total infected", "currently infected", "symptomatic",
#              "quarantined", "hospitalized", "recovered", "vaccinated", "transmitter"
#              "deceased"
property = "deceased"

# Show the results
#plot_all_strategies(results1, results2, results3, results4, results5, timesteps, property)
