import pickle as pkl
import pandas as pd
import numpy as np
import os
from ..data.openeye import load_openeye_sdf, load_openeye_pdb, get_ligand_rmsd_from_pdb_and_sdf, split_openeye_mol
from openeye import oechem

SCORE_COLUMNS = ["RMSD", "POSIT_R", "Chemgauss4", "MCSS_Rank"]


class DockingDataset():

    def __init__(self, pkl_fn, dir_path):
        self.pkl_fn = pkl_fn
        self.dir_path = dir_path

        for path in [self.pkl_fn, self.dir_path]:
            assert os.path.exists(path)

    def read_pkl(self):
        self.compound_ids, self.xtal_ids, self.res_ranks = pkl.load(open(self.pkl_fn, 'rb'))

    def get_cmpd_dir_path(self, cmpd_id):
        ## make sure this directory exists
        cmpd_dir = os.path.join(self.dir_path, cmpd_id)
        print(cmpd_dir)
        # assert os.path.exists(cmpd_dir)
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
            sdf_list = [fn for fn in fn_list if os.path.splitext(fn)[1] == '.sdf']

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

                except:

                    ## however its a convenient way of identifying which is the original xtal
                    ref_xtal = xtal
                    # ref_sdf_fn = f"{ref_xtal}_{chain}/{ref_xtal}_{chain}.sdf"
                    ref_pdb_fn = f"{ref_xtal}_{chain}/{ref_xtal}_{chain}_bound.pdb"

                    ## save the ref filename to the dictionary and make the mcss_rank -1
                    # ref_fn_dict[cmpd_id] = ref_sdf_fn
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
            {"Complex_ID": cmplx_ids,
             "Compound_ID": cmpd_ids,
             "Crystal_ID": xtal_ids,
             "Chain_ID": chain_ids,
             "MCSS_Rank": mcss_ranks,
             "SDF_Filename": sdf_fns,
             "Reference": ref_fns
             }
        )

    def calculate_rmsd_and_posit_score(self, fragalysis_dir):
        rmsds = []
        posit_scores = []
        docking_scores = []
        print(self.df.head().to_dict(orient='index'))
        for data in self.df.to_dict(orient='index').values():
            cmpd_id = data["Compound_ID"]
            sdf_fn = data["SDF_Filename"]
            ref_fn = data["Reference"]

            cmpd_dir = self.get_cmpd_dir_path(cmpd_id)

            sdf_path = os.path.join(cmpd_dir, sdf_fn)
            ref_path = os.path.join(fragalysis_dir, ref_fn)

            print(f"Loading rmsd calc on {sdf_path} compared to {ref_path}")

            docking_results = get_ligand_rmsd_from_pdb_and_sdf(ref_path,
                                                               mobile_path=sdf_path,
                                                               fetch_docking_results=True)
            posit_scores.append(docking_results['posit'])
            docking_scores.append(docking_results['chemgauss'])
            rmsds.append(docking_results['rmsd'])

        self.df["POSIT"] = posit_scores
        self.df["Chemgauss4"] = docking_scores
        self.df["RMSD"] = rmsds

    def write_csv(self, output_csv_fn):
        self.df.to_csv(output_csv_fn, index=False)

    def analyze_docking_results(self,
                                fragalysis_dir,
                                output_csv_fn,
                                test=False):
        self.organize_docking_results()
        # self.to_df()
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
        raise NotImplementedError(f"good score acquisition not implemented for {score}")
    return lambda_func


class DockingResults():
    """
    This is a class to parse docking results from a csv file.
    Mainly for mainipulating the data in various useful ways.
    """

    def __init__(self, csv_path):
        ## load in data and replace the annoying `-1.0` and `-1` values with nans
        self.df = pd.read_csv(csv_path).replace(-1.0, np.nan).replace(-1, np.nan)

    def get_grouped_df(self,
                            groupby_ID_column="Compound_ID",
                            score_columns=SCORE_COLUMNS):
        score_df_list = []
        for score in score_columns:
            if not score in self.df.columns:
                print(f"Skipping {score}")
                continue
            if not self.df[score].any():
                print(f"Skipping {score} since no non-NA scores were found")
                continue
            not_na = self.df.groupby(groupby_ID_column)[[score]].count()
            good = self.df.groupby(groupby_ID_column)[[score]].apply(get_good_score(score))
            mean = self.df.groupby(groupby_ID_column)[[score]].mean()
            min = self.df.groupby(groupby_ID_column)[[score]].min()
            # if sum(not_na[not_na != 0]) == 0:
            #     print(f"Skipping {score} since no non-NA scores were found")
            feature_df = pd.concat([not_na, good, mean, min], axis=1)
            feature_df.columns = [f"{name}_{score}" for name in ["Not_NA", "Good", "Mean", "Min"]]
            score_df_list.append(feature_df)
        grouped_df = pd.concat(score_df_list, axis=1)
        grouped_df[groupby_ID_column] = grouped_df.index
        return grouped_df

    def get_compound_df(self, **kwargs):
        self.compound_df = self.get_grouped_df(groupby_ID_column="Compound_ID", **kwargs)

    def get_structure_df(self, **kwargs):
        self.structure_df = self.get_grouped_df(groupby_ID_column="Structure_Source", **kwargs)
        with open("../scripts/mers_structures.csv") as f:
            mers_structure_df = pd.read_csv(f)
        self.structure_df["Resolution"] = list(mers_structure_df.Resolution)

    def get_best_structure_per_compound(self,
                                        filter_score = "RMSD",
                                        filter_value = 2.5,
                                        score_order=["Chemgauss4", "RMSD"]):
        """
        A fancier implementation of this might first just filter the structures based on certain scores and *then*
        pick the best one from that set, but I haven't done that yet.

        Parameters
        ----------
        score_order: this will allow us to pick the structures by filtering in the order that this is given

        Returns
        -------

        """
        ## First filter by RMSD

        best_df = self.df[self.df[filter_score] < 2.5]

        for score in score_order:

            ## first find the minimum value of `score` for each Compound_ID
            min_value = best_df.groupby("Compound_ID")[score].min()

            ## add merge that to every row in the original dataframe
            best_df = best_df.merge(min_value, on="Compound_ID", suffixes=('', '_min'))

            ## then only keep the rows where the value of `score` is the minimum one
            best_df = best_df[best_df[score] == best_df[f"{score}_min"]]

            ## since that might not result in only one `Complex_ID` for each `Compound_ID`, iterate through the scores
            ## could add a check so that we don't iterate through the scores
        self.best_df = best_df



