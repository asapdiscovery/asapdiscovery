import pickle as pkl
import pandas as pd
import numpy as np
import os, re
from ..data.openeye import (
    get_ligand_rmsd_from_pdb_and_sdf,
)


class DockingDataset:
    def __init__(self, pkl_fn, dir_path):
        self.pkl_fn = pkl_fn
        self.dir_path = dir_path

        for path in [self.pkl_fn, self.dir_path]:
            assert os.path.exists(path)

    def read_pkl(self):
        self.compound_ids, self.xtal_ids, self.res_ranks = pkl.load(
            open(self.pkl_fn, "rb")
        )

    def get_cmpd_dir_path(self, cmpd_id):
        ## make sure this directory exists
        cmpd_dir = os.path.join(self.dir_path, cmpd_id)
        assert os.path.exists(cmpd_dir)
        return cmpd_dir

    def organize_docking_results(self):
        ## Make lists we want to use to collect information about docking results
        cmpd_ids = []
        xtal_ids = []
        chain_ids = []
        mcss_ranks = []
        sdf_fns = []
        cmplx_ids = []

        ## dictionary of reference sdf files for each compound id
        ref_fn_dict = {}

        ## since each compound has its own directory, we can use that directory
        for cmpd_id in self.compound_ids:
            cmpd_id = cmpd_id.rstrip()

            cmpd_dir = self.get_cmpd_dir_path(cmpd_id)

            ## get list of files in this directory
            fn_list = os.listdir(cmpd_dir)

            ## Process this list into info
            ## TODO: use REGEX instead
            sdf_list = [
                fn for fn in fn_list if os.path.splitext(fn)[1] == ".sdf"
            ]

            ## For each sdf file in this list, get all the information
            ## This is very fragile to the file naming scheme
            for fn in sdf_list:
                info = fn.split(".")[0].split("_")
                xtal = info[3]
                chain = info[4]
                cmpd_id = info[7]

                ## the ligand re-docked to their original xtal have no mcss rank, so this will fail
                try:
                    mcss_rank = info[9]

                except IndexError:

                    ## however its a convenient way of identifying which is the original xtal
                    ref_xtal = xtal
                    ref_pdb_fn = (
                        f"{ref_xtal}_{chain}/{ref_xtal}_{chain}_bound.pdb"
                    )

                    ## save the ref filename to the dictionary and make the mcss_rank -1
                    ref_fn_dict[cmpd_id] = ref_pdb_fn
                    mcss_rank = -1

                ## append all the stuff to lists!
                cmpd_ids.append(cmpd_id)
                xtal_ids.append(xtal)
                chain_ids.append(chain)
                mcss_ranks.append((mcss_rank))
                sdf_fns.append(fn)
                cmplx_ids.append(f"{cmpd_id}_{xtal}_{chain}_{mcss_rank}")

        ref_fns = [ref_fn_dict[cmpd_id] for cmpd_id in cmpd_ids]

        self.df = pd.DataFrame(
            {
                "Complex_ID": cmplx_ids,
                "Compound_ID": cmpd_ids,
                "Crystal_ID": xtal_ids,
                "Chain_ID": chain_ids,
                "MCSS_Rank": mcss_ranks,
                "SDF_Filename": sdf_fns,
                "Reference": ref_fns,
            }
        )

    def calculate_rmsd_and_posit_score(self, fragalysis_dir):
        rmsds = []
        posit_scores = []
        docking_scores = []
        print(self.df.head().to_dict(orient="index"))
        for data in self.df.to_dict(orient="index").values():
            cmpd_id = data["Compound_ID"]
            sdf_fn = data["SDF_Filename"]
            ref_fn = data["Reference"]

            cmpd_dir = self.get_cmpd_dir_path(cmpd_id)

            sdf_path = os.path.join(cmpd_dir, sdf_fn)
            ref_path = os.path.join(fragalysis_dir, ref_fn)

            print(f"Loading rmsd calc on {sdf_path} compared to {ref_path}")

            docking_results = get_ligand_rmsd_from_pdb_and_sdf(
                ref_path, mobile_path=sdf_path, fetch_docking_results=True
            )
            posit_scores.append(docking_results["posit"])
            docking_scores.append(docking_results["chemgauss"])
            rmsds.append(docking_results["rmsd"])

        self.df["POSIT"] = posit_scores
        self.df["Chemgauss4"] = docking_scores
        self.df["RMSD"] = rmsds

    def write_csv(self, output_csv_fn):
        self.df.to_csv(output_csv_fn, index=False)

    def analyze_docking_results(
        self, fragalysis_dir, output_csv_fn, test=False
    ):
        self.organize_docking_results()

        if test:
            self.df = self.df.head()
        self.calculate_rmsd_and_posit_score(fragalysis_dir)
        self.write_csv(output_csv_fn=output_csv_fn)


def get_good_score(score):
    """
    The idea here is that given an array from a DataFrame, x, that is a particular score type, there should be a
    single function that will tell you whether we think this score is "good" or not. I'm not sure this is the best way
    to do this but someway to make it obvious when we are using which cutoff would be good.

    Parameters
    ----------
    score

    Returns
    -------

    """
    if score == "RMSD":
        lambda_func = lambda x: x[(x <= 2.5)].count()
    elif score == "POSIT":
        lambda_func = lambda x: x[(x > 0.7)].count()
    elif score == "Chemgauss4":
        lambda_func = lambda x: x[(x < 0)].count()
    elif score == "POSIT_R":
        lambda_func = lambda x: x[(x < 0.3)].count()
    else:
        raise NotImplementedError(
            f"good score acquisition not implemented for {score}. "
            f'Try using one of ["RMSD", "POSIT", "Chemgauss4", "POSIT_R"]'
        )
    return lambda_func


class DockingResults:
    """
    This is a class to parse docking results from a csv file.
    Mainly for mainipulating the data in various useful ways.
    """

    def __init__(self, csv_path):
        ## load in data and replace the annoying `-1.0` and `-1` values with nans
        self.df = (
            pd.read_csv(csv_path).replace(-1.0, np.nan).replace(-1, np.nan)
        )

    def get_grouped_df(
        self,
        groupby_ID_column="Compound_ID",
        score_columns=["RMSD", "POSIT_R", "Chemgauss4", "MCSS_Rank"],
    ):
        """
        The purpose of this function is to get a dataframe with meaningful information grouped by either the Compound_ID
        or by the Structure_Source.

        Parameters
        ----------
        groupby_ID_column
        score_columns

        Returns
        -------

        """
        # TODO: Default argument `score_columns` is a mutable. This can lead to unexpected behavior in Python.
        # TODO: I think this can be done without a for loop by replacing [[score]] with [score_columns]
        score_df_list = []
        for score in score_columns:
            if not score in self.df.columns:
                print(f"Skipping {score}")
                continue
            if not self.df[score].any():
                print(f"Skipping {score} since no non-NA scores were found")
                continue
            # Group by the groupby_ID_column and then get either the number, mean, or mean of the identified score
            not_na = self.df.groupby(groupby_ID_column)[[score]].count()
            mean = self.df.groupby(groupby_ID_column)[[score]].mean()
            min = self.df.groupby(groupby_ID_column)[[score]].min()

            # I barely understand how this works but basically it
            # applies the lambda function returned by `get_good_score` and then groups the results, which means that
            # it can count how many "good" scores there were for each group.
            good = self.df.groupby(groupby_ID_column)[[score]].apply(
                get_good_score(score)
            )

            feature_df = pd.concat([not_na, good, mean, min], axis=1)
            feature_df.columns = [
                f"{score}_{name}" for name in ["Not_NA", "Good", "Mean", "Min"]
            ]
            score_df_list.append(feature_df)
        grouped_df = pd.concat(score_df_list, axis=1)
        grouped_df[groupby_ID_column] = grouped_df.index
        return grouped_df

    def get_compound_df(self, csv_path=False, **kwargs):
        if csv_path:
            self.compound_df = pd.read_csv(csv_path)
        else:
            self.compound_df = self.get_grouped_df(
                groupby_ID_column="Compound_ID", **kwargs
            )

    def get_structure_df(self, csv_path=False, resolution_csv=None, **kwargs):
        """
        Either pull the structure_df from the csv file or generate it using the get_grouped_df function
        In addition to what the function normally does it also adds the resolution

        Parameters
        ----------
        csv_path
        kwargs

        Returns
        -------

        """
        if csv_path:
            self.structure_df = pd.read_csv(csv_path)
        else:
            self.structure_df = self.get_grouped_df(
                groupby_ID_column="Structure_Source", **kwargs
            )
            if resolution_csv:
                ## Get a dictionary with {"PDB ID": "Resolution", ..}
                with open(resolution_csv) as f:
                    resolution_df = pd.read_csv(f)
                    resolution_df.index = resolution_df["PDB ID"]
                    resolution_dict = resolution_df.to_dict()["Resolution"]

                ## Iterates over the dataframe
                ## Uses regex to pull out the PDB ID from each Structure_Source
                ## and makes a list of the Resolutions that is saved to a column
                ## of the new structure_df
                self.structure_df["Resolution"] = [
                    resolution_dict.get(
                        re.search(
                            pattern=r"[\d][A-Za-z0-9]{3}",
                            string=pdb,
                        ).group(0)
                    )
                    for pdb in self.structure_df.Structure_Source.to_list()
                ]

    def get_best_structure_per_compound(
        self,
        filter_score="RMSD",
        filter_value=2.5,
        score_order=["POSIT_R", "Chemgauss4", "RMSD"],
        score_ascending=[True, True, True],
    ):
        """
        Gets the best structure by first filtering based on the filter_score and filter_value,
        then sorts in order of the scores listed in score_order.

        As with everything else, lower scores are assumed to be better, requiring a conversion of some scores.

        Parameters
        ----------
        filter_score
        filter_value
        score_order

        Returns
        -------

        """
        # TODO: also this is really fragile
        # TODO: default argument `score_order` is a mutable. This can lead to unexpected behavior in python.
        ## first do filtering
        print(filter_score, filter_value)
        if filter_score and type(filter_value) == float:
            print(f"Filtering by {filter_score} less than {filter_value}")
            filtered_df = self.df[self.df[filter_score] < filter_value]
            sort_list = ["Compound_ID"] + score_order

        ## sort dataframe, ascending (smaller / better scores will move to the top)
        sorted_df = filtered_df.sort_values(
            sort_list, ascending=[True, True, True, True]
        )

        ## group by compound id and return the top row for each group
        g = sorted_df.groupby("Compound_ID")
        self.best_df = g.head(1)

        return self.best_df

    def write_dfs_to_csv(self, output_dir):
        self.df.to_csv(
            os.path.join(output_dir, "all_results_cleaned.csv"), index=False
        )
        self.compound_df.to_csv(
            os.path.join(output_dir, "by_compound.csv"), index=False
        )
        self.structure_df.to_csv(
            os.path.join(output_dir, "by_structure.csv"), index=False
        )
        self.best_df.to_csv(
            os.path.join(output_dir, "best_results.csv"), index=False
        )


def load_dataframes(input_dir):
    """
    Load csv files from an input directory

    Parameters
    ----------
    input_dir

    Returns
    -------
    dictionary of dataframes

    """
    best_results_csv = os.path.join(input_dir, "best_results.csv")
    all_results_csv = os.path.join(input_dir, "all_results_cleaned.csv")
    by_compound_csv = os.path.join(input_dir, "by_compound.csv")
    by_structure_csv = os.path.join(input_dir, "by_structure.csv")

    df = pd.read_csv(all_results_csv)
    df.index = df.Complex_ID
    tidy = df.melt(id_vars="Complex_ID")
    df = df.round({"Chemgauss4": 3, "POSIT": 3, "POSIT_R": 3, "RMSD": 3})

    by_compound = pd.read_csv(by_compound_csv)
    by_compound_tidy = by_compound.melt(id_vars="Compound_ID")

    by_structure = pd.read_csv(by_structure_csv)
    by_structure_tidy = by_structure.melt(id_vars="Structure_Source")

    return {
        "tidy": tidy,
        "df": df,
        "by_compound_tidy": by_compound_tidy,
        "by_compound": by_compound,
        "by_structure_tidy": by_structure_tidy,
        "by_structure": by_structure,
    }


def filter_df_by_two_columns(
    df,
    xaxis_column_name,
    yaxis_column_name,
    x_range=None,
    y_range=None,
):
    """
    Simple function for filtering a dataframe by the values of particular columns

    Parameters
    ----------
    df
    xaxis_column_name: str
    yaxis_column_name: str
    x_range: [min, max]
    y_range: [min, max

    Returns
    -------

    """
    if not x_range:
        x_range = (df[xaxis_column_name].min(), df[xaxis_column_name].max())
    if not y_range:
        y_range = (df[yaxis_column_name].min(), df[yaxis_column_name].max())

    dff = df[
        (df[xaxis_column_name] > x_range[0])
        & (df[xaxis_column_name] < x_range[1])
        & (df[yaxis_column_name] > y_range[0])
        & (df[yaxis_column_name] < y_range[1])
    ]
    return dff
