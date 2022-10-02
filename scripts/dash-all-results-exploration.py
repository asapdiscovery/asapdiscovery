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
                            # min=tidy["variable"]["value"].min(),
                            # max=tidy[
                            #     tidy["variable"] == "crossfilter-xaxis-column"
                            # ]["value"].max(),
                            # step=0.1,
                            # marks={0: "0", 1: "1", 3.5: "3.5"},
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
                        "float": "top",
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
                            # max=tidy[
                            #     tidy["variable"] == "crossfilter-xaxis-column"
                            # ]["value"].max(),
                            # step=0.1,
                            # marks={0: "0", 1: "1", 3.5: "3.5"},
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
        # html.Div(
        #     [
        #         dcc.Graph(
        #             id="cross-filter-indicator-contour",
        #             # hoverData={"points": [{"customdata": "Japan"}]},
        #         )
        #     ],
        #     style={
        #         # "width": "49%",
        #         "display": "inline-block",
        #         # "float": "right",
        #         "padding": "0 20",
        #     },
        # ),
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
                "width": "49%",
                "display": "inline-block",
                "float": "right",
                # "padding": "0 20",
            },
        ),
        # html.Div(
        #     [
        #         dcc.Graph(
        #             id="crossfilter-indicator-contour",
        #             # hoverData={"points": [{"customdata": "Japan"}]},
        #         )
        #     ],
        #     style={
        #         # "width": "49%",
        #         "display": "inline-block",
        #         # "float": "right",
        #         "padding": "0 20",
        #     },
        # ),
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
def update_graph(
    xaxis_column_name,
    yaxis_column_name,
    xaxis_type,
    yaxis_type,
    x_range,
    y_range,
    color_column,
):
    # x = tidy[tidy["variable"] == xaxis_column_name]["value"]
    # x_filtered = x[(x > x_range[0]) & (x < x_range[1])]
    # y = tidy[tidy["variable"] == yaxis_column_name]["value"]
    # y_filtered = y[(y > y_range[0]) & (y < y_range[1])]
    #
    # final = tidy[tidy["variable"] == xaxis_column_name]["value"]
    #
    # filtered = tidy[
    #     ([tidy["variable"] == xaxis_column_name]["value"] > x_range[0])
    #     # & ([tidy["variable"] == xaxis_column_name]["value"] < x_range[1])
    #     # & ([tidy["variable"] == yaxis_column_name]["value"] > y_range[0])
    #     # & ([tidy["variable"] == yaxis_column_name]["value"] < y_range[1])
    # ]

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
        # hover_name=dff[dff["variable"] == yaxis_column_name][
        #     "Country Name"
        # ],
        hover_data=["Complex_ID"],
        color=color_column,
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


app.run_server(port=9001, debug=True)
