import sys, os, argparse

# TODO: Do we need to add plotly to our environment yaml?
import plotly.express as px

sys.path.append(
    f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
)
from covid_moonshot_ml.docking import DockingResults


def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-d",
        "--input_dir",
        required=True,
        type=str,
        help="Path to input directory containing CSVs cleaned by clean_results_csv.py.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        type=str,
        help="Path to output directory",
    )

    return parser.parse_args()


def main():
    args = get_args()

    assert os.path.exists(args.input_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    assert os.path.exists(args.output_dir)

    ## Get File Paths
    all_results = os.path.join(args.input_dir, "all_results_cleaned.csv")
    by_compound = os.path.join(args.input_dir, "by_compound.csv")
    by_structure = os.path.join(args.input_dir, "by_compound.csv")

    ## Load Results
    dr = DockingResults(csv_path=all_results)
    dr.get_compound_df(csv_path=by_compound)
    dr.get_structure_df(csv_path=by_structure)

    features = [
        feature
        for feature in dr.structure_df.columns
        if not feature == "Compound_ID"
    ]

    ## make per_structure figures
    df = dr.structure_df
    for feature in features:
        fig = px.bar(df.sort_values(feature), y=feature, text_auto=".2s")
        file_path = os.path.join(
            args.output_dir, f"per_structure_{feature}.png"
        )
        fig.write_image(file_path)

    ## make per_compound figures
    df = dr.compound_df
    for feature in features:
        fig = px.histogram(df.sort_values(feature), x=feature, text_auto=".2s")
        file_path = os.path.join(args.output_dir, f"per_compound_{feature}.png")
        fig.write_image(file_path)


if __name__ == "__main__":
    main()
