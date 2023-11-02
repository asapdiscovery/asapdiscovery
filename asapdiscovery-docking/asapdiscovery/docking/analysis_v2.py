"""
The purpose of this module is to provide a set of standard functionality to analyze the docking results.
"""
# from asapdiscovery.data.schema_v2 import complex, ligand, target
from asapdiscovery.docking.docking_data_validation import DockingResultCols
import pandas as pd


class DockingResults:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # self.complexes = self.get_complexes()
        # self.ligands = self.get_ligands()
        # self.targets = self.get_targets()
        self.docking_result_cols = DockingResultCols
        self.score_columns = self.get_score_columns()

    @classmethod
    def from_csv(cls, csv_path: str) -> "DockingResults":
        """
        Loads a DockingResults object from a CSV file
        """
        df = pd.read_csv(csv_path, index_col=0)

        # probably validation methods should go here

        return cls(df)

    # def get_complexes(self) -> list[complex]:
    #     """
    #     Returns the list of complexes in the docking results
    #     """
    #     return self.df[self.DockingResultCols.DU_STRUCTURE.value].unique()
    #
    # def get_ligands(self) -> list[ligand]:
    #     """
    #     Returns the list of ligands in the docking results
    #     """
    #     return self.df[self.DockingResultCols.LIGAND_ID.value].unique()
    #
    # def get_targets(self) -> list[target]:
    #     """
    #     Returns the list of targets in the docking results
    #     """
    #     return self.df[self.DockingResultCols.TARGET_ID.value].unique()

    def get_score_columns(self) -> list[str]:
        """
        Returns the list of score columns in the docking results
        """
        return [col for col in self.df.columns if col.startswith("docking-score-")]


def calculate_rmsd_stats(
    df: pd.DataFrame,
    query_mol_id: str,  # Column name of the molecule ID
    reference_selection: str,
    score_column: str,
    group_by: [str],
    cumulative=True,
    ref_structure_stride: int = 10,
    ref_structure_id: str = "Structure_Name",
    n_bootstraps: int = 3,
    fraction_structures_used: float = 1.0,
    rmsd_col="RMSD",
    rmsd_cutoff: float = 2.0,
    count_nrefs=False,
):
    dfs = []
    for i in range(n_bootstraps):

        # Randomize the order of the structures
        randomized = df.sample(frac=1)

        for n_ref in range(
            1, len(randomized[ref_structure_id].unique()), ref_structure_stride
        ):
            # Get subset of structures bassed on reference selection method
            if reference_selection == "random":
                if cumulative:
                    subset_df = randomized.groupby([query_mol_id] + group_by).head(
                        n_ref
                    )
                else:
                    _range = range(n_ref, n_ref + ref_structure_stride)
                    subset_df = randomized.groupby([query_mol_id] + group_by).nth(
                        _range
                    )
            else:
                # first sort by the reference selection method
                if cumulative:
                    subset_df = (
                        randomized.sort_values(reference_selection)
                        .groupby([query_mol_id] + group_by)
                        .head(n_ref)
                    )
                else:
                    _range = range(n_ref, n_ref + ref_structure_stride)
                    subset_df = (
                        randomized.sort_values(reference_selection)
                        .groupby([query_mol_id] + group_by)
                        .nth(_range)
                    )
            # Rank the poses by score
            scored_df = (
                subset_df.sort_values(score_column)
                .groupby([query_mol_id] + group_by)
                .head(1)
            )
            rmsd_stats_series = scored_df.groupby(group_by, group_keys=True)[
                rmsd_col
            ].apply(lambda x: x <= rmsd_cutoff).groupby(group_by).sum() / len(
                df[query_mol_id].unique()
            )

            split_cols_list = []
            score_list = []
            n_references = []

            min_nrefs = []
            max_nrefs = []
            mean_nrefs = []

            if count_nrefs:
                nref_data = (
                    subset_df.groupby([query_mol_id] + group_by)[score_column]
                    .count()
                    .groupby(group_by)
                    .describe()
                )

            for split_col in rmsd_stats_series.index:
                split_cols_list.append(split_col)
                score_list.append(rmsd_stats_series[split_col])
                n_references.append(n_ref)

                if count_nrefs:
                    min_nrefs.append(nref_data["min"][split_col])
                    max_nrefs.append(nref_data["max"][split_col])
                    mean_nrefs.append(nref_data["mean"][split_col])

            # n_allowed_refs = n_references if cumulative else ref_structure_stride

            # TODO: Replace all these hard-coded names with
            return_df = pd.DataFrame(
                {
                    "Fraction": score_list,
                    "Version": split_cols_list,
                    "Number of References": n_references,
                    "Mean Number of References": mean_nrefs if count_nrefs else None,
                    "Max Number of References": max_nrefs if count_nrefs else None,
                    "Min Number of References": min_nrefs if count_nrefs else None,
                    "Structure_Split": reference_selection,
                }
            )
            if reference_selection == "random":
                return_df["Split_Value_min"] = "Random"
                return_df["Split_Value_max"] = "Random"
            else:
                return_df["Split_Value_min"] = subset_df[reference_selection].min()
                return_df["Split_Value_max"] = subset_df[reference_selection].max()
            dfs.append(return_df)

    combined = pd.concat(dfs)
    return combined
