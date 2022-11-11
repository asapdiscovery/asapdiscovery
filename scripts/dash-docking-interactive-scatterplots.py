"""
Plot interactive scatter plots where clicking on one changes the data that is shown on the others

"""
from dash import html, Input, Output
import argparse, os, sys

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from covid_moonshot_ml.docking.analysis import load_dataframes
from covid_moonshot_ml.plotting import plotting, plotly_dash_functions


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
            plotly_dash_functions.get_basic_plot(
                id="crossfilter-indicator-contour"
            ),
            plotly_dash_functions.get_heading(
                id="by-structure-header", text="Filtered by Structure"
            ),
            plotly_dash_functions.get_heading(
                id="by-structure-header", text="Filtered by Compound"
            ),
            plotly_dash_functions.get_basic_plot(id="by-structure-scatterplot"),
            plotly_dash_functions.get_basic_plot(id="by-compound-scatterplot"),
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
    def update_plot(*args):
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

    ## Create a by_structure filtered scatterplot
    ## Use dash decorator syntax to pass arguments into the scatterplot function
    @app.callback(
        Output("by-structure-scatterplot", "figure"),
        Input("crossfilter-xaxis-column", "value"),
        Input("crossfilter-yaxis-column", "value"),
        Input("crossfilter-xaxis-type", "value"),
        Input("crossfilter-yaxis-type", "value"),
        Input("x-axis-slider", "value"),
        Input("y-axis-slider", "value"),
        Input("crossfilter-color", "value"),
        Input("full-scatterplot", "clickData"),
    )
    def update_plot(*args):
        clickData = args[-1]
        print(clickData)
        complex_ID = clickData["points"][0]["customdata"][0]
        print(complex_ID)
        ## Get Compound
        structure = df.loc[complex_ID, "Structure_Source"][0]
        print(structure)
        ## Filter by compound
        dff = df[df["Structure_Source"] == structure]
        fig = plotting.scatter_plot(dff, *args[0:-1])
        fig.update_layout(title=f"{structure}")
        return fig

    ## Create a by_compound filtered scatterplot
    ## Use dash decorator syntax to pass arguments into the scatterplot function
    @app.callback(
        Output("by-compound-scatterplot", "figure"),
        Input("crossfilter-xaxis-column", "value"),
        Input("crossfilter-yaxis-column", "value"),
        Input("crossfilter-xaxis-type", "value"),
        Input("crossfilter-yaxis-type", "value"),
        Input("x-axis-slider", "value"),
        Input("y-axis-slider", "value"),
        Input("crossfilter-color", "value"),
        Input("full-scatterplot", "clickData"),
    )
    def update_plot(*args):
        clickData = args[-1]
        print(clickData)
        complex_ID = clickData["points"][0]["customdata"][0]
        print(complex_ID)
        ## Get Compound
        compound = df.loc[complex_ID, "Compound_ID"][0]
        print(compound)
        ## Filter by compound
        dff = df[df["Compound_ID"] == compound]

        fig = plotting.scatter_plot(dff, *args[0:-1])
        fig.update_layout(title=f"{compound}")
        return fig

    ## Run the server!
    app.run_server(port=9001, debug=True)


if __name__ == "__main__":
    main()
