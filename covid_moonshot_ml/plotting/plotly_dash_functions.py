# TODO: put dash html logic here so that it can be loaded into the dash-docking-exploration scripts
import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table, ctx
import plotly.express as px
import json, argparse, os


def get_dash_app():
    app = Dash(__name__)
    styles = {"pre": {"border": "thin lightgrey solid", "overflowX": "scroll"}}
    return app, styles


# def wrap_plotting_function(app)


def get_contour_plot():
    return html.Div(
        [
            dcc.Graph(
                id="crossfilter-indicator-contour",
            )
        ],
        style={
            "display": "inline-block",
            "padding": "0 20",
            # "float": "right",
        },
    )


def get_filters(variable_list):
    return html.Div(
        [
            html.Div(
                [
                    html.H4("X-Axis"),
                    dcc.Dropdown(
                        variable_list,  ## the whole list
                        variable_list[0],  ## the default
                        id="crossfilter-xaxis-column",
                    ),
                    html.P("Filter X-Axis:"),
                    dcc.RangeSlider(
                        id="x-axis-slider",
                        min=-50,
                        max=50,
                        value=[-50, 50],
                    ),
                    dcc.RadioItems(
                        ["Linear", "Log"],
                        "Linear",
                        id="crossfilter-xaxis-type",
                        labelStyle={
                            "display": "inline-block",
                            "marginTop": "5px",
                        },
                    ),
                ],
                style={
                    "width": "49%",
                    "display": "inline-block",
                },
            ),
            html.Div(
                [
                    html.H4("Y-Axis"),
                    dcc.Dropdown(
                        variable_list,  ## the whole list
                        variable_list[-1],  ## the default
                        id="crossfilter-yaxis-column",
                    ),
                    html.P("Filter Y-Axis:"),
                    dcc.RangeSlider(
                        id="y-axis-slider",
                        min=0,
                        max=30,
                        value=[0, 30],
                    ),
                    dcc.RadioItems(
                        ["Linear", "Log"],
                        "Linear",
                        id="crossfilter-yaxis-type",
                        labelStyle={
                            "display": "inline-block",
                            "marginTop": "5px",
                        },
                    ),
                ],
                style={
                    "width": "49%",
                    "display": "inline-block",
                },
            ),
        ],
        style={
            "padding": "10px 5px",
        },
    )
