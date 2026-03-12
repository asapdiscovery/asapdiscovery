import os
import pickle as pkl
import re
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from asapdiscovery.data.backend.openeye import oechem, oeshape
from asapdiscovery.data.schema.ligand import Ligand


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
        # make sure this directory exists
        cmpd_dir = os.path.join(self.dir_path, cmpd_id)
        assert os.path.exists(cmpd_dir)
        return cmpd_dir

    def organize_docking_results(self):
        # Make lists we want to use to collect information about docking results
        cmpd_ids = []
        xtal_ids = []
        chain_ids = []
        mcss_ranks = []
        sdf_fns = []
        cmplx_ids = []

        # dictionary of reference sdf files for each compound id
        ref_fn_dict = {}

        # since each compound has its own directory, we can use that directory
        for cmpd_id in self.compound_ids:
            cmpd_id = cmpd_id.rstrip()

            cmpd_dir = self.get_cmpd_dir_path(cmpd_id)

            # get list of files in this directory
            fn_list = os.listdir(cmpd_dir)

            # Process this list into info
            # TODO: use REGEX instead
            sdf_list = [fn for fn in fn_list if os.path.splitext(fn)[1] == ".sdf"]

            # For each sdf file in this list, get all the information
            # This is very fragile to the file naming scheme
            for fn in sdf_list:
                info = fn.split(".")[0].split("_")
                xtal = info[3]
                chain = info[4]
                cmpd_id = info[7]

                # the ligand re-docked to their original xtal have no mcss rank, so this
                # will fail
                try:
                    mcss_rank = info[9]

                except IndexError:
                    # however its a convenient way of identifying which is the original
                    # xtal
                    ref_xtal = xtal
                    ref_pdb_fn = f"{ref_xtal}_{chain}/{ref_xtal}_{chain}_bound.pdb"

                    # save the ref filename to the dictionary and make the mcss_rank -1
                    ref_fn_dict[cmpd_id] = ref_pdb_fn
                    mcss_rank = -1

                # append all the stuff to lists!
                cmpd_ids.append(cmpd_id)
                xtal_ids.append(xtal)
                chain_ids.append(chain)
                mcss_ranks.append(mcss_rank)
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

    def write_csv(self, output_csv_fn):
        self.df.to_csv(output_csv_fn, index=False)

    def analyze_docking_results(self, fragalysis_dir, output_csv_fn, test=False):
        self.organize_docking_results()

        if test:
            self.df = self.df.head()
        self.write_csv(output_csv_fn=output_csv_fn)


def get_good_score(score):
    """
    The idea here is that given an array from a DataFrame, x, that is a particular score
    type, there should be a single function that will tell you whether we think this
    score is "good" or not. I'm not sure this is the best way to do this but someway to
    make it obvious when we are using which cutoff would be good.

    Parameters
    ----------
    score

    Returns
    -------

    """
    if score == "RMSD":
        lambda_func = lambda x: x[(x <= 2.5)].count()  # noqa: E731
    elif score == "POSIT":
        lambda_func = lambda x: x[(x > 0.7)].count()  # noqa: E731
    elif score == "Chemgauss4":
        lambda_func = lambda x: x[(x < 0)].count()  # noqa: E731
    elif score == "POSIT_R":
        lambda_func = lambda x: x[(x < 0.3)].count()  # noqa: E731
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

    column_names_dict = {
        "legacy": [
            "ligand_id",
            "du_structure",
            "docked_file",
            "docked_RMSD",
            "POSIT_prob",
            "chemgauss4_score",
            "clash",
        ],
        "legacy_smiles": [
            "ligand_id",
            "du_structure",
            "docked_file",
            "docked_RMSD",
            "POSIT_prob",
            "chemgauss4_score",
            "clash",
            "SMILES",
        ],
        "legacy_cleaned": [
            "Compound_ID",
            "Structure_Source",
            "Docked_File",
            "RMSD",
            "POSIT",
            "Chemgauss4",
            "Clash",
        ],
        "legacy_cleaned_dimer": [
            "Compound_ID",
            "Structure_Source",
            "Dimer",
            "Docked_File",
            "RMSD",
            "POSIT",
            "Chemgauss4",
            "Clash",
        ],
        "default_dimer": [
            "Compound_ID",
            "Structure_Source",
            "Dimer",
            "Docked_File",
            "RMSD",
            "POSIT",
            "Chemgauss4",
            "Clash",
            "SMILES",
        ],
        "default": [
            "Compound_ID",
            "Structure_Source",
            "Docked_File",
            "RMSD",
            "POSIT",
            "Chemgauss4",
            "Clash",
            "SMILES",
        ],
    }

    def __init__(self, csv_path=None, df=None, column_names="default"):
        """

        Parameters
        ----------
        csv_path: path to csv file
            Optional
        df: pd.DataFrame
        """
        if type(csv_path) is str:
            # load in data and replace the annoying `-1.0` and `-1` values with nans
            self.df = pd.read_csv(csv_path).replace(-1.0, np.nan).replace(-1, np.nan)
        elif isinstance(df, pd.DataFrame):
            if self.column_names_dict.get(column_names):
                df.columns = self.column_names_dict.get(column_names)
            self.df = df

        else:
            raise Exception("Must pass either a dataframe or a csv path")

    def get_grouped_df(
        self,
        groupby_ID_column="Compound_ID",
        score_columns=("RMSD", "POSIT_R", "Chemgauss4", "MCSS_Rank"),
    ):
        """
        The purpose of this function is to get a dataframe with meaningful information
        grouped by either the Compound_ID or by the Structure_Source.

        Parameters
        ----------
        groupby_ID_column
        score_columns

        Returns
        -------

        """
        # TODO: Default argument `score_columns` is a mutable. This can lead to
        # unexpected behavior in Python.
        # TODO: I think this can be done without a for loop by
        # replacing [[score]] with [score_columns]
        score_df_list = []
        for score in score_columns:
            if not score in self.df.columns:  # noqa: E713
                print(f"Skipping {score}")
                continue
            if not self.df[score].any():
                print(f"Skipping {score} since no non-NA scores were found")
                continue
            # Group by the groupby_ID_column and then get either the number, mean,
            # or mean of the identified score
            not_na = self.df.groupby(groupby_ID_column)[[score]].count()
            mean = self.df.groupby(groupby_ID_column)[[score]].mean()
            min = self.df.groupby(groupby_ID_column)[[score]].min()

            # I barely understand how this works but basically it
            # applies the lambda function returned by `get_good_score` and then groups
            # the results, which means that
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

    def get_compound_df(self, csv_path=None, **kwargs):
        if csv_path:
            self.compound_df = pd.read_csv(csv_path)
        else:
            self.compound_df = self.get_grouped_df(
                groupby_ID_column="Compound_ID", **kwargs
            )

    def get_structure_df(self, csv_path=None, resolution_csv=None, **kwargs):
        """
        Either pull the structure_df from the csv file or generate it using the
        get_grouped_df function in addition to what the function normally does it also
        adds the resolution

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
                # Get a dictionary with {"PDB ID": "Resolution", ..}
                with open(resolution_csv) as f:
                    resolution_df = pd.read_csv(f)
                    resolution_df.index = resolution_df["PDB ID"]
                    resolution_dict = resolution_df.to_dict()["Resolution"]

                # Iterates over the dataframe
                # Uses regex to pull out the PDB ID from each Structure_Source
                # and makes a list of the Resolutions that is saved to a column
                # of the new structure_df
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
        score_order=("POSIT_R", "Chemgauss4", "RMSD"),
        score_ascending=(True, True, True),
    ):
        """
        Gets the best structure by first filtering based on the filter_score and
        filter_value, then sorts in order of the scores listed in score_order.

        As with everything else, lower scores are assumed to be better, requiring a
        conversion of some scores.

        Parameters
        ----------
        filter_score
        filter_value
        score_order

        Returns
        -------

        """
        # TODO: also this is really fragile
        # first do filtering
        print(filter_score, filter_value)
        if filter_score and type(filter_value) is float:
            print(f"Filtering by {filter_score} less than {filter_value}")
            filtered_df = self.df[self.df[filter_score] < filter_value]
            sort_list = ["Compound_ID"] + score_order

        # sort dataframe, ascending (smaller / better scores will move to the top)
        sorted_df = filtered_df.sort_values(
            sort_list, ascending=[True] + score_ascending
        )

        # group by compound id and return the top row for each group
        g = sorted_df.groupby("Compound_ID")
        self.best_df = g.head(1)

        return self.best_df

    def write_dfs_to_csv(self, output_dir):
        self.df.to_csv(os.path.join(output_dir, "all_results_cleaned.csv"), index=False)
        self.compound_df.to_csv(
            os.path.join(output_dir, "by_compound.csv"), index=False
        )
        self.structure_df.to_csv(
            os.path.join(output_dir, "by_structure.csv"), index=False
        )
        self.best_df.to_csv(os.path.join(output_dir, "best_results.csv"), index=False)


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


def calculate_rmsd_openeye(
    reference_ligand: oechem.OEMol or oechem.OEGraphMol,
    docked_ligand: oechem.OEMol or oechem.OEGraphMol,
) -> float:
    """
    Calculate the root-mean-square deviation (RMSD) between the coordinates of two OpenEye molecules. The OEMol or
    OEGraphMol objects are converted to coordinates using the .GetCoords() method and then oechem.OERMSD is used.

    Parameters
    ----------
    reference_ligand : oechem.OEMol or oechem.OEGraphMol
        The reference molecule to which the docked molecule is compared.
    docked_ligand : oechem.OEMol or oechem.OEGraphMol
        The docked molecule to be compared to the reference molecule.

    Returns
    -------
    float
        The RMSD between the two molecules' coordinates, with hydrogens excluded.
    """
    # Calculate RMSD
    oechem.OECanonicalOrderAtoms(reference_ligand)
    oechem.OECanonicalOrderBonds(reference_ligand)
    oechem.OECanonicalOrderAtoms(docked_ligand)
    oechem.OECanonicalOrderBonds(docked_ligand)
    # Get coordinates, filtering out Hs
    predocked_coords = [
        c
        for a in reference_ligand.GetAtoms()
        for c in reference_ligand.GetCoords()[a.GetIdx()]
        if a.GetAtomicNum() != 1
    ]
    docked_coords = [
        c
        for a in docked_ligand.GetAtoms()
        for c in docked_ligand.GetCoords()[a.GetIdx()]
        if a.GetAtomicNum() != 1
    ]
    rmsd = oechem.OERMSD(
        oechem.OEDoubleArray(predocked_coords),
        oechem.OEDoubleArray(docked_coords),
        len(predocked_coords) // 3,
    )
    return rmsd


class TanimotoType(str, Enum):
    """
    Enum for the different types of Tanimoto coefficients that can be calculated.
    """

    SHAPE = "TanimotoShape"
    COLOR = "TanimotoColor"
    COMBO = "TanimotoCombo"


def calculate_tanimoto_oe(
    refmol: Ligand,
    fitmol: Ligand,
    compute_type: TanimotoType = TanimotoType.COMBO,
):
    """
    Calculate the Tanimoto coefficient between two molecules using OpenEye's shape toolkit.

    Parameters
    ----------
    refmol : Ligand
        The reference molecule to which the docked molecule is compared.
    fitmol : Ligand
        The docked molecule to be compared to the reference molecule.

    Returns
    -------
    float
        The Tanimoto coefficient between the two molecules.
    """
    refmol = refmol.to_oemol()
    fitmol = fitmol.to_oemol()

    # Prepare reference molecule for calculation
    # With default options this will remove any explicit hydrogens present
    prep = oeshape.OEOverlapPrep()
    prep.Prep(refmol)

    # Get appropriate function to calculate exact shape
    shapeFunc = oeshape.OEOverlapFunc()
    shapeFunc.SetupRef(refmol)

    res = oeshape.OEOverlapResults()
    prep.Prep(fitmol)
    shapeFunc.Overlap(fitmol, res)
    if compute_type == TanimotoType.SHAPE:
        return res.GetShapeTanimoto()
    elif compute_type == TanimotoType.COLOR:
        return res.GetColorTanimoto()
    elif compute_type == TanimotoType.COMBO:
        return res.GetTanimotoCombo()


def write_all_rmsds_to_reference(
    ref_mol: oechem.OEGraphMol,
    docked_mols: [oechem.OEGraphMol],
    output_dir: Path,
    compound_id,
):
    """
    Calculates the RMSD between a reference molecule and a list of docked molecules,
    and saves the results to a .npy file.

    Parameters
    ----------
    ref_mol: oechem.OEGraphMol
        The reference molecule, for which the RMSD is calculated.
    docked_mols: [oechem.OEGraphMol]
        A list of docked molecules, for which the RMSD is calculated with respect to the reference molecule.
    output_dir: Path
        The directory where the output file will be saved.
    compound_id: str
        A unique identifier for the compound being analyzed, used as the name of the output file.

    Returns
    -------
    None
    """
    rmsds = [
        (docked_mol.GetTitle(), calculate_rmsd_openeye(ref_mol, docked_mol))
        for docked_mol in docked_mols
    ]

    np.save(str(output_dir / f"{compound_id}.npy"), rmsds)
