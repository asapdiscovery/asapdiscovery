import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table, ctx
import plotly.express as px
import json, argparse, os

parser = argparse.ArgumentParser(description="")
## Input arguments
parser.add_argument(
    "-i",
    "--input_dir",
    required=True,
    help="Path to directory containing docking csvs.",
)
args = parser.parse_args()

all_results_csv = os.path.join(args.input_dir, "all_results_cleaned.csv")
by_compound_csv = os.path.join(args.input_dir, "by_compound.csv")
by_structure_csv = os.path.join(args.input_dir, "by_structure.csv")

app = Dash(__name__)

df = pd.read_csv(all_results_csv)
df.index = df.Complex_ID
tidy = df.melt(id_vars="Complex_ID")
df = df.round({"Chemgauss4": 3, "POSIT": 3, "POSIT_R": 3, "RMSD": 3})

by_compound = pd.read_csv(by_compound_csv)
by_compound_tidy = by_compound.melt(id_vars="Compound_ID")

by_structure = pd.read_csv(by_structure_csv)
by_structure_tidy = by_structure.melt(id_vars="Structure_Source")

styles = {"pre": {"border": "thin lightgrey solid", "overflowX": "scroll"}}

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H4("X-Axis"),
                        dcc.Dropdown(
                            tidy["variable"].unique(),
                            "Chemgauss4",
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
                            tidy["variable"].unique(),
                            "RMSD",
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
        ),
        html.Div(
            [
                html.H4("Color"),
                dcc.Dropdown(
                    tidy["variable"].unique(),
                    "POSIT",
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
        html.Div(
            [
                dcc.Graph(
                    id="by-compound",
                )
            ],
            style={
                "display": "inline-block",
                "padding": "0 20",
                # "float": "right",
            },
        ),
        html.Div(
            [
                dcc.Graph(
                    id="by-structure",
                )
            ],
            style={
                "display": "inline-block",
                "padding": "0 20",
                # "float": "right",
            },
        ),
        html.Div(
            [
                html.H4("By Structure (Counts)"),
                dcc.Graph(id="per-structure-bar-chart"),
            ],
            style={"width": "49%", "display": "inline-block"},
        ),
        html.Div(
            [
                html.H4("By Structure (Values)"),
                dcc.Graph(id="per-structure-bar-chart2"),
            ],
            style={"width": "49%", "display": "inline-block"},
        ),
        dash_table.DataTable(
            df.to_dict("records"),
            id="table",
            columns=[
                {"name": column, "id": column}
                for column in [
                    "Compound_ID",
                    "Structure_Source",
                    "RMSD",
                    "POSIT",
                    "Chemgauss4",
                ]
            ],
            style_table={
                "width": "50%",
                "float": "center",
                "display": "inline-block",
                "padding": "0 20",
                # "float": "right",
            },
            filter_action="custom",
            filter_query="",
        ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    [
                        dcc.Markdown(
                            """
                            **Click Data**
    
                            Click on points in the graph.
                        """
                        ),
                        html.Pre(id="click-data", style=styles["pre"]),
                    ],
                    className="three columns",
                ),
            ],
        ),
    ]
)


@app.callback(
    Output("click-data", "children"),
    Input("crossfilter-indicator-scatter", "clickData"),
)
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)


def filter_scatterplot(xaxis_column_name, yaxis_column_name, x_range, y_range):
    dff = df[
        (df[xaxis_column_name] > x_range[0])
        & (df[xaxis_column_name] < x_range[1])
        & (df[yaxis_column_name] > y_range[0])
        & (df[yaxis_column_name] < y_range[1])
    ]
    return dff


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
    filtered = filter_scatterplot(
        xaxis_column_name, yaxis_column_name, x_range, y_range
    )

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
)
def update_contour(
    xaxis_column_name,
    yaxis_column_name,
    xaxis_type,
    yaxis_type,
    x_range,
    y_range,
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


@app.callback(
    Output("table", "data"), Input("crossfilter-indicator-scatter", "clickData")
)
def update_table(clickData):
    if clickData:
        complex_ID = clickData["points"][0]["customdata"][0]

        ## Get Compound
        compound = df.loc[complex_ID, "Compound_ID"]

    else:
        compound = df["Compound_ID"][0]

    ## Filter by compound
    dff = df[df["Compound_ID"] == compound]

    return dff.to_dict("records")


# Input("crossfilter-indicator-scatter", "clickData")


@app.callback(
    Output("by-compound", "figure"),
    Input("table", "data"),
    Input("crossfilter-xaxis-column", "value"),
    Input("crossfilter-yaxis-column", "value"),
    Input("crossfilter-xaxis-type", "value"),
    Input("crossfilter-yaxis-type", "value"),
    Input("crossfilter-color", "value"),
)
def update_filtered_scatter(
    data_dict,
    xaxis_column_name,
    yaxis_column_name,
    xaxis_type,
    yaxis_type,
    color_column,
):
    filtered = pd.DataFrame(data_dict)
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
    Output("by-structure", "figure"),
    Input("crossfilter-indicator-scatter", "clickData"),
    Input("by-compound", "clickData"),
    Input("crossfilter-xaxis-column", "value"),
    Input("crossfilter-yaxis-column", "value"),
    Input("crossfilter-xaxis-type", "value"),
    Input("crossfilter-yaxis-type", "value"),
    Input("x-axis-slider", "value"),
    Input("y-axis-slider", "value"),
    Input("crossfilter-color", "value"),
)
def update_by_structure(
    clickData1,
    clickData2,
    xaxis_column_name,
    yaxis_column_name,
    xaxis_type,
    yaxis_type,
    x_range,
    y_range,
    color_column,
):
    filtered = filter_scatterplot(
        xaxis_column_name, yaxis_column_name, x_range, y_range
    )
    input_source = ctx.triggered_id
    print(input_source)
    if input_source in [
        "crossfilter-indicator-scatter",
        "by-compound",
    ]:
        click_data = ctx.triggered[0]["value"]
        complex_ID = click_data["points"][0]["customdata"][0]
    else:
        complex_ID = filtered["Complex_ID"][0]

    ## Get Structure
    structure = filtered.loc[complex_ID, "Structure_Source"]

    ## Filter by structure
    dff = filtered
    dff.loc[:, "Selection"] = filtered["Structure_Source"] == structure

    fig = px.scatter(
        dff.sort_values(["Selection"]),
        x=xaxis_column_name,
        y=yaxis_column_name,
        hover_data=["Complex_ID"],
        opacity=0.5,
        color="Selection",
        color_discrete_sequence=["grey", "blue"],
    )
    fig.update_layout(
        legend_title=f"Structure_Source: {structure}",
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
    Output("per-structure-bar-chart", "figure"),
    Input("crossfilter-indicator-scatter", "clickData"),
    Input("by-compound", "clickData"),
)
def per_structure_bar_chart(clickData1, clickData2):

    ## Get "Count" Columns"
    count_columns = [
        column
        for column in by_structure.columns
        if "Good" in column or "Not_NA" in column
    ]

    input_source = ctx.triggered_id
    print(input_source)
    if not input_source or input_source in [
        "crossfilter-indicator-scatter",
        "by-compound",
    ]:
        if not input_source:
            complex_ID = df["Complex_ID"][0]
        else:
            click_data = ctx.triggered[0]["value"]
            complex_ID = click_data["points"][0]["customdata"][0]

        ## Get Structure
        structure = df.loc[complex_ID, "Structure_Source"]

        ## Filter by structure
        dff = by_structure_tidy[
            by_structure_tidy["Structure_Source"] == structure
        ]
        dff = dff[dff["variable"].isin(count_columns)]
        # dff.loc[:, "Selection"] = (
        #     by_structure_tidy["Structure_Source"] == structure
        # )

        fig = px.bar(
            dff,
            x="value",
            y="Structure_Source",
            # hover_data=["Resolution"],
            color="variable",
            barmode="group",
            height=800,
            width=800,
        )
        return fig


@app.callback(
    Output("per-structure-bar-chart2", "figure"),
    Input("crossfilter-indicator-scatter", "clickData"),
    Input("by-compound", "clickData"),
)
def per_structure_bar_chart(clickData1, clickData2):

    ## Get "Values" Columns"
    count_columns = [
        column
        for column in by_structure.columns
        if "Min" in column or "Mean" in column
    ]

    input_source = ctx.triggered_id
    print(input_source)
    if not input_source or input_source in [
        "crossfilter-indicator-scatter",
        "by-compound",
    ]:
        if not input_source:
            complex_ID = df["Complex_ID"][0]
        else:
            click_data = ctx.triggered[0]["value"]
            complex_ID = click_data["points"][0]["customdata"][0]

        ## Get Structure
        structure = df.loc[complex_ID, "Structure_Source"]

        ## Filter by structure
        dff = by_structure_tidy[
            by_structure_tidy["Structure_Source"] == structure
        ]
        dff = dff[dff["variable"].isin(count_columns)]
        # dff.loc[:, "Selection"] = (
        #     by_structure_tidy["Structure_Source"] == structure
        # )

        fig = px.bar(
            dff,
            x="value",
            y="Structure_Source",
            # hover_data=["Resolution"],
            color="variable",
            barmode="group",
            height=800,
            width=800,
        )
        return fig


app.run_server(port=9001, debug=True)
