"""
Plot interactive scatter plots where clicking on one changes the data that is shown on the others

"""
import argparse
import json
import os
import sys

import pandas as pd
from dash import Input, Output, dcc, html

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from asapdiscovery.dataviz import plotly_dash_functions, plotting
from asapdiscovery.docking.analysis import load_dataframes


def get_args():
    parser = argparse.ArgumentParser(description="")
    ## Input arguments
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Path to directory containing docking csvs.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    df_dict = load_dataframes(args.input_dir)
    tidy = df_dict["tidy"]
    df = df_dict["df"]

    ## Get Dash App
    app, styles = plotly_dash_functions.get_dash_app()

    ## Make contour plot
    app.layout = html.Div(
        [
            plotly_dash_functions.get_filters(
                variable_list=tidy["variable"].unique(),
                default_x="Chemgauss4",
                default_y="RMSD",
                xrange=[-30, 10],
                yrange=[0, 8],
            ),
            plotly_dash_functions.get_color_variable(
                variable_list=tidy.variable.unique()
            ),
            plotly_dash_functions.get_heading(
                id="full-scatterplot-header", text="Full Scatterplot"
            ),
            plotly_dash_functions.get_heading(
                id="full-scatterplot-header", text="Contour Plot"
            ),
            plotly_dash_functions.get_basic_plot(id="full-scatterplot"),
            plotly_dash_functions.get_basic_plot(id="crossfilter-indicator-contour"),
            plotly_dash_functions.get_heading(
                id="by-structure-header", text="Filtered by Structure"
            ),
            plotly_dash_functions.get_heading(
                id="by-structure-header", text="Filtered by Compound"
            ),
            plotly_dash_functions.get_basic_plot(id="by-structure-scatterplot"),
            plotly_dash_functions.get_basic_plot(id="by-compound-scatterplot"),
            dcc.Store(id="selected-structure"),
            dcc.Store(id="by-structure-filtered-df"),
            dcc.Store(id="selected-compound"),
            dcc.Store(id="by-compound-filtered-df"),
        ]
    )

    ## Use dash decorator syntax to pass arguments into the scatterplot function
    @app.callback(
        Output("full-scatterplot", "figure"),
        Input("crossfilter-xaxis-column", "value"),
        Input("crossfilter-yaxis-column", "value"),
        Input("crossfilter-xaxis-type", "value"),
        Input("crossfilter-yaxis-type", "value"),
        Input("x-axis-slider", "value"),
        Input("y-axis-slider", "value"),
        Input("crossfilter-color", "value"),
    )
    def update_scatterplot(*args):
        fig = plotting.scatter_plot(df, *args)
        return fig

    ## Use dash decorator syntax to pass arguments into the update_contour function
    @app.callback(
        Output("crossfilter-indicator-contour", "figure"),
        Input("crossfilter-xaxis-column", "value"),
        Input("crossfilter-yaxis-column", "value"),
        Input("crossfilter-xaxis-type", "value"),
        Input("crossfilter-yaxis-type", "value"),
        Input("x-axis-slider", "value"),
        Input("y-axis-slider", "value"),
    )
    def update_contour(*args):
        print(*args)
        fig = plotting.contour_plot(df, *args)
        return fig

    ## Instead of processing the dataframe separately in every scatterplot/table function
    ## this way I can process it every time I click and then get everything from that
    @app.callback(
        Output("selected-structure", "data"),
        Output("selected-compound", "data"),
        Output("by-structure-filtered-df", "data"),
        Output("by-compound-filtered-df", "data"),
        Input("full-scatterplot", "clickData"),
    )
    def process_click_data(clickData):
        if clickData:
            compound = clickData["points"][0]["customdata"][1]
            structure = clickData["points"][0]["customdata"][2]
        else:
            structure = df.Structure_Source[0]
            compound = df.Compound_ID[0]

        sdf = df[df["Structure_Source"] == structure]
        cdf = df[df["Compound_ID"] == compound]

        return (
            json.dumps(structure),
            json.dumps(compound),
            sdf.to_json(orient="split"),
            cdf.to_json(orient="split"),
        )

    ## Create a by_structure filtered scatterplot
    ## Use dash decorator syntax to pass arguments into the scatterplot function
    @app.callback(
        Output("by-structure-scatterplot", "figure"),
        Input("by-structure-filtered-df", "data"),
        Input("selected-structure", "data"),
        Input("crossfilter-xaxis-column", "value"),
        Input("crossfilter-yaxis-column", "value"),
        Input("crossfilter-xaxis-type", "value"),
        Input("crossfilter-yaxis-type", "value"),
        Input("x-axis-slider", "value"),
        Input("y-axis-slider", "value"),
        Input("crossfilter-color", "value"),
    )
    def update_plot(df_data, structure, *args):
        df = pd.read_json(df_data, orient="split")
        structure = json.loads(structure)
        fig = plotting.scatter_plot(df, *args)
        fig.update_layout(title=structure)
        return fig

    ## Create a by_compound filtered scatterplot
    ## Use dash decorator syntax to pass arguments into the scatterplot function
    @app.callback(
        Output("by-compound-scatterplot", "figure"),
        Input("by-compound-filtered-df", "data"),
        Input("selected-compound", "data"),
        Input("crossfilter-xaxis-column", "value"),
        Input("crossfilter-yaxis-column", "value"),
        Input("crossfilter-xaxis-type", "value"),
        Input("crossfilter-yaxis-type", "value"),
        Input("x-axis-slider", "value"),
        Input("y-axis-slider", "value"),
        Input("crossfilter-color", "value"),
    )
    def update_plot(df_data, compound, *args):
        df = pd.read_json(df_data, orient="split")
        compound = json.loads(compound)
        fig = plotting.scatter_plot(df, *args)
        fig.update_layout(title=compound)
        return fig

    ## Run the server!
    app.run_server(port=9001, debug=True)


if __name__ == "__main__":
    main()
