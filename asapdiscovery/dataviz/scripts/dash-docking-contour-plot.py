"""
The goal of this script is to test importing functions and html logic from other places
to make a simple dash html instance that is straightforward to read.

The goal is to replace the dash-docking-exploration and dash-all-results-exploration 
with scripts that look like this one.

This script currently takes plotly functions from covid_moonshot_ml.plotting.plotting
and dash html logic from plotly_dash_functions.

Adding new functions would require 3 steps (as far as I understand it): 
1. making a plotly figure creator function
2. making the appropriate html logic function
3. using the decorator syntax to appropriately pass inputs from the html logic to the plotly creator
"""
from dash import html, Input, Output
import argparse, os, sys

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from asapdiscovery.docking.analysis import load_dataframes
from asapdiscovery.dataviz import plotting, plotly_dash_functions


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
    tidy, df, by_compound_tidy, by_structure_tidy = load_dataframes(
        args.input_dir
    )

    ## Get Dash App
    app, styles = plotly_dash_functions.get_dash_app()

    ## Make contour plot
    app.layout = html.Div(
        [
            plotly_dash_functions.get_basic_plot(
                id="crossfilter-indicator-contour"
            ),
            plotly_dash_functions.get_filters(tidy.variable.unique()),
        ]
    )

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

    ## Run the server!
    app.run_server(port=9001, debug=True)


if __name__ == "__main__":
    main()
