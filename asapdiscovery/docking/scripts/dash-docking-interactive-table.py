"""
This gives an example of how to add a very basic interactive table
"""
from dash import html
import argparse, os, sys

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from asapdiscovery.docking.analysis import load_dataframes
from asapdiscovery.docking import plotly_dash_functions


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

    print(df.head().to_dict("records"))
    ## Make contour plot
    app.layout = html.Div(
        [
            plotly_dash_functions.get_dash_table(
                id="table",
                variable_dict=df.to_dict("records"),
                columns=[
                    {"name": column, "id": column}
                    for column in [
                        "Complex_ID",
                        "Compound_ID",
                        "Structure_Source",
                        "RMSD",
                        "POSIT",
                        "Chemgauss4",
                        "Dimer",
                    ]
                ],
            ),
        ]
    )

    ## Run the server!
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
