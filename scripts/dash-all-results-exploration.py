import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

app = Dash(__name__)

df = pd.read_csv(
    "/Volumes/Rohirrim/local_test/mers_hallucination_hybrid/posit_hybrid_no_relax/all_results_cleaned.csv"
)
tidy = df.melt(id_vars="Complex_ID")

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H4("X-Axis"),
                        dcc.Dropdown(
                            tidy["variable"].unique(),
                            "POSIT_R",
                            id="crossfilter-xaxis-column",
                        ),
                        html.P("Filter X-Axis:"),
                        dcc.RangeSlider(
                            id="x-axis-slider",
                            min=-100,
                            max=100,
                            value=[0, 3],
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
                            tidy["variable"].unique(),
                            "RMSD",
                            id="crossfilter-yaxis-column",
                        ),
                        html.P("Filter Y-Axis:"),
                        dcc.RangeSlider(
                            id="y-axis-slider",
                            min=-100,
                            max=100,
                            value=[0, 3],
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
        ),
        html.Div(
            [
                html.H4("Color"),
                dcc.Dropdown(
                    tidy["variable"].unique(),
                    "POSIT_R",
                    id="crossfilter-color",
                ),
            ],
            style={
                "width": "98%",
                "display": "inline-block",
                # "float": "left",
            },
        ),
        html.Div(
            [
                dcc.Graph(
                    id="crossfilter-indicator-scatter",
                )
            ],
            style={
                "display": "inline-block",
                "padding": "0 20",
                # "float": "left",
            },
        ),
        html.Div(
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
        ),
    ]
)


@app.callback(
    Output("crossfilter-indicator-scatter", "figure"),
    Input("crossfilter-xaxis-column", "value"),
    Input("crossfilter-yaxis-column", "value"),
    Input("crossfilter-xaxis-type", "value"),
    Input("crossfilter-yaxis-type", "value"),
    Input("x-axis-slider", "value"),
    Input("y-axis-slider", "value"),
    Input("crossfilter-color", "value"),
)
def update_scatter(
    xaxis_column_name,
    yaxis_column_name,
    xaxis_type,
    yaxis_type,
    x_range,
    y_range,
    color_column,
):
    filtered = df[
        (df[xaxis_column_name] > x_range[0])
        & (df[xaxis_column_name] < x_range[1])
        & (df[yaxis_column_name] > y_range[0])
        & (df[yaxis_column_name] < y_range[1])
    ]

    fig = px.scatter(
        filtered,
        x=xaxis_column_name,
        y=yaxis_column_name,
        hover_data=["Complex_ID"],
        color=color_column,
        color_continuous_scale="dense",
    )

    fig.update_xaxes(
        title=xaxis_column_name,
        type="linear" if xaxis_type == "Linear" else "log",
    )

    fig.update_yaxes(
        title=yaxis_column_name,
        type="linear" if yaxis_type == "Linear" else "log",
    )

    fig.update_layout(
        margin={"l": 40, "b": 40, "t": 40, "r": 40}, hovermode="closest"
    )

    return fig


@app.callback(
    Output("crossfilter-indicator-contour", "figure"),
    Input("crossfilter-xaxis-column", "value"),
    Input("crossfilter-yaxis-column", "value"),
    Input("crossfilter-xaxis-type", "value"),
    Input("crossfilter-yaxis-type", "value"),
    Input("x-axis-slider", "value"),
    Input("y-axis-slider", "value"),
    Input("crossfilter-color", "value"),
)
def update_contour(
    xaxis_column_name,
    yaxis_column_name,
    xaxis_type,
    yaxis_type,
    x_range,
    y_range,
    color_column,
):
    filtered = df[
        (df[xaxis_column_name] > x_range[0])
        & (df[xaxis_column_name] < x_range[1])
        & (df[yaxis_column_name] > y_range[0])
        & (df[yaxis_column_name] < y_range[1])
    ]

    fig = px.density_contour(
        filtered,
        x=xaxis_column_name,
        y=yaxis_column_name,
        marginal_x="histogram",
        marginal_y="histogram",
    )
    fig.update_traces(
        contours_coloring="heatmap",
        selector=dict(type="histogram2dcontour"),
        colorscale="Peach",
    )
    fig.update_xaxes(
        title=xaxis_column_name,
        type="linear" if xaxis_type == "Linear" else "log",
    )

    fig.update_yaxes(
        title=yaxis_column_name,
        type="linear" if yaxis_type == "Linear" else "log",
    )

    fig.update_layout(
        margin={"l": 40, "b": 40, "t": 40, "r": 40}, hovermode="closest"
    )

    return fig


app.run_server(port=9001, debug=True)
