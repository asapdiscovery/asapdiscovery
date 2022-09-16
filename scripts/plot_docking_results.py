import sys, os, argparse
import plotly.express as px
sys.path.append(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}')
from covid_moonshot_ml.docking.analysis import DockingResults

def get_args():
    parser = argparse.ArgumentParser(description='')

    ## Input arguments
    parser.add_argument('-i', "--input_csv", required=True,
        help='Path to CSV file containing docking results.')
    parser.add_argument('-o', "--output_dir", required=True,
                        help="Path to output directory")

    return parser.parse_args()

def main():
    args = get_args()

    assert os.path.exists(args.input_csv)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    assert os.path.exists(args.output_dir)

    dr = DockingResults(args.input_csv)

    ## this is a bunch of csv ingesting that is very dependent on the way the csv looks
    dr.df = dr.df.drop("Unnamed: 0", axis=1)
    dr.df["MERS_structure"] = dr.df["MERS_structure"].apply(lambda x: x.split("_")[1])

    ## Rename the columns, add POSIT_R and Complex_ID since those are useful
    dr.df.columns = ["Compound_ID", "Structure_Source", "Docked_File", "RMSD", "POSIT", "Chemgauss4", "Clash"]
    dr.df['POSIT_R'] = 1 - dr.df.POSIT
    dr.df["Complex_ID"] = dr.df.Compound_ID + "_" + dr.df.Structure_Source

    ## Clean the Docked_File paths because there are extra `/`
    ## also, some of the file paths are NaNs so we need to only keep the ones that are strings
    cleaned = [directory
               for string in dr.df.Docked_File if type(string) == str
               for directory in string.split("/") if not len(directory) == 0
               ]
    dr.df.Docked_File = "/".join(cleaned)

    ## Re-sort the dataframe by the Compound_ID so that its nice and alphabetical and re-index based on that
    dr.df = dr.df.sort_values(["Compound_ID"]).reset_index(drop=True)

    ## See the code for more details!
    dr.get_compound_df()
    dr.get_structure_df()

    features = [feature for feature in dr.structure_df.columns if
                feature.split("_")[0] in ["Not", "Good", "Mean", "Min"]]

    ## make per_structure figures
    df = dr.structure_df
    for feature in features:
        fig = px.bar(df.sort_values(feature),
                     y=feature,
                     text_auto=".2s")
        file_path = os.path.join(args.output_dir, f"per_structure_{feature}.png")
        fig.write_image(file_path)

    ## make per_compound figures
    df = dr.compound_df
    for feature in features:
        fig = px.histogram(df.sort_values(feature),
                           x=feature,
                           text_auto=".2s")
        file_path = os.path.join(args.output_dir, f"per_compound_{feature}.png")
        fig.write_image(file_path)

    ## get the best structure per compound
    dr.get_best_structure_per_compound()
    file_path = os.path.join(args.output_dir, f"mers_fauxalysis.csv")
    dr.best_df.to_csv(file_path, index=False)

if __name__ == '__main__':
    main()