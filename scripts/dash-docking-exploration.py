import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import os, argparse

app = Dash(__name__)


parser = argparse.ArgumentParser(description="")
## Input arguments
parser.add_argument(
    "-i",
    "--input_dir",
    required=True,
    help="Path to directory containing docking csvs.",
)
args = parser.parse_args()

by_compound_csv = os.path.join(args.input_dir, "by_compound.csv")
by_structure_csv = os.path.join(args.input_dir, "by_structure.csv")

by_compound = pd.read_csv(by_compound_csv)
by_compound_tidy = by_compound.melt(id_vars="Compound_ID")

by_structure = pd.read_csv(by_structure_csv)
by_structure_tidy = by_structure.melt(id_vars="Structure_Source")

app.layout = html.Div(
    [
        html.Div(
            [
                html.H4("By Structure"),
                dcc.Dropdown(
                    by_structure.columns,
                    "Resolution",
                    id="bar-xaxis",
                ),
                dcc.Graph(id="per-structure-bar-chart"),
                html.P("Filter by Resolution:"),
                dcc.RangeSlider(
                    id="range-slider",
                    min=0,
                    max=3.5,
                    step=0.1,
                    marks={0: "0", 1: "1", 3.5: "3.5"},
                    value=[0.5, 3.5],
                ),
            ],
            style={"width": "49%", "display": "inline-block"},
        ),
        html.Div(
            [
                html.H4("By Compound"),
                dcc.Dropdown(
                    by_compound.columns,
                    "POSIT_R_Good",
                    id="cmpd-bar-xaxis",
                ),
                dcc.Graph(id="per-compound-bar-chart"),
            ],
            style={"width": "49%", "display": "inline-block", "float": "right"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H4("X-Axis"),
                        dcc.Dropdown(
                            by_structure_tidy["variable"].unique(),
                            "Resolution",
                            id="crossfilter-xaxis-column",
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
                        "float": "top",
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    [
                        html.H4("Y-Axis"),
                        dcc.Dropdown(
                            by_structure_tidy["variable"].unique(),
                            "POSIT_R_Good",
                            id="crossfilter-yaxis-column",
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
                        "float": "bottom",
                        "display": "inline-block",
                    },
                ),
            ],
            style={
                "padding": "10px 5px",  # "float": "left",
                # "display": "inline-block",
            },
        ),
        html.Div(
            [
                dcc.Graph(
                    id="crossfilter-indicator-scatter",
                    # hoverData={"points": [{"customdata": "Japan"}]},
                )
            ],
            style={
                # "width": "49%",
                "display": "inline-block",
                # "float": "right",
                "padding": "0 20",
            },
        ),
    ]
)


@app.callback(
    Output("per-structure-bar-chart", "figure"),
    Input("range-slider", "value"),
    Input("bar-xaxis", "value"),
)
def per_structure_bar_chart(slider_range, x_variable):
    low, high = slider_range
    mask = (
        (by_structure["Resolution"] > low) & (by_structure["Resolution"] < high)
    ) | (by_structure["Resolution"].isna())
    filtered_df = by_structure[mask]
    fig = px.bar(
        filtered_df.sort_values(x_variable),
        x=x_variable,
        y="Structure_Source",
        hover_data=["Resolution"],
        height=800,
        width=800,
    )
    return fig


@app.callback(
    Output("per-compound-bar-chart", "figure"),
    # Input("range-slider", "value"),
    Input("cmpd-bar-xaxis", "value"),
)
def per_compound_bar_chart(x_variable):
    fig = px.bar(
        by_compound.sort_values(x_variable),
        x=x_variable,
        y="Compound_ID",
        height=800,
        width=800,
    )
    return fig


@app.callback(
    Output("crossfilter-indicator-scatter", "figure"),
    Input("crossfilter-xaxis-column", "value"),
    Input("crossfilter-yaxis-column", "value"),
    Input("crossfilter-xaxis-type", "value"),
    Input("crossfilter-yaxis-type", "value"),
)
def update_graph(
    xaxis_column_name,
    yaxis_column_name,
    xaxis_type,
    yaxis_type,
):
    dff = by_structure_tidy

    fig = px.scatter(
        x=dff[dff["variable"] == xaxis_column_name]["value"],
        y=dff[dff["variable"] == yaxis_column_name]["value"],
        # hover_name=dff[dff["variable"] == yaxis_column_name][
        #     "Country Name"
        # ],
        hover_data={
            "Resolution (Å)": dff[dff["variable"] == "Resolution"]["value"],
            "Structure": dff["Structure_Source"].unique(),
        },
        # color=dff[dff["variable"] == "Resolution"]["value"],
        # color=dff["Structure_Source"].unique(),
    )

    # fig.update_traces(
    #     customdata=dff[dff["variable"] == yaxis_column_name][
    #         "Country Name"
    #     ]
    # )

    fig.update_xaxes(
        title=xaxis_column_name,
        type="linear" if xaxis_type == "Linear" else "log",
    )

    fig.update_yaxes(
        title=yaxis_column_name,
        type="linear" if yaxis_type == "Linear" else "log",
    )

    fig.update_layout(
        margin={"l": 40, "b": 40, "t": 10, "r": 0}, hovermode="closest"
    )

    return fig


app.run_server(port=9000, debug=True)
