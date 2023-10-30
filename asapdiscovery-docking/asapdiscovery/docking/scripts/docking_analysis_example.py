from asapdiscovery.docking import analysis
from asapdiscovery.data.schema_v2 import complex, ligand, target
import numpy as np, pandas as pd

# load in data containing info about docking results
dr = analysis.DockingResults.from_csv("docking_results.csv")

# load in information about molecules / targets if not present in docking results data,
# and merge with docking results data
# TODO not sure how this should look
ligands: [ligand] = dr.GetLigands()
targets: [target] = dr.GetTargets()

# Now we should be able to ask some questions about the data
# For instance, how does the fraction of ligands with RMSD < 2.0 depend on the number of references used, for
# different ways of selecting references?

# By random
random_results_df = dr.CalculateRMSDStats(
    rmsd_cutoff=2.0,
    reference_selections="random",
    n_references=range(
        1, len(targets) + 1, 5  # Use 1, 6, 11, etc references until all are used
    ),
    n_bootstraps=100,
    group_by=["Version"],  # Group the statistics by this list of columns
)

random_results_df.head()
"""
n_references    fraction_mean   fraction_std    version
1               0.43            0.15            All   
6               0.57            0.05            All
11              0.68            0.05            All
16              0.71            0.05            All    
"""

# By date
date_results_df = dr.CalculateRMSDStats(
    rmsd_cutoff=2.0,
    reference_selections="Date",
    n_references=range(
        1, len(targets) + 1, 5  # Use 1, 6, 11, etc references until all are used
    ),
    n_bootstraps=100,
    group_by=["Version"],  # Group the statistics by this list of columns
)


# What does CalculateRMSDStats actually need to do?


def calculate_rmsd_stats(
    df: pd.DataFrame,
    query_mol_id: str,  # Column name of the molecule ID
    reference_selection: str,
    score_column: str,
    group_by: [str],
    ref_structure_stride: int = 10,
    ref_structure_id: str = "Structure_Name",
    n_bootstraps: int = 3,
    fraction_structures_used: float = 1.0,
    rmsd_col="RMSD",
    rmsd_cutoff: float = 2.0,
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
                subset_df = randomized.groupby([query_mol_id] + group_by).head(n_ref)
            else:
                # first sort by the reference selection method
                subset_df = (
                    randomized.sort_values(reference_selection)
                    .groupby([query_mol_id] + group_by)
                    .head(n_ref)
                )
            print(subset_df.head())
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

            for split_col in rmsd_stats_series.index:
                split_cols_list.append(split_col)
                score_list.append(rmsd_stats_series[split_col])
                n_references.append(n_ref)

            return_df = pd.DataFrame(
                {
                    "Fraction": score_list,
                    "Version": split_cols_list,
                    "Number of References": n_references,
                }
            )
            return_df["Split_Value"] = subset_df.Structure_Date.max()
            dfs.append(return_df)

    combined = pd.concat(dfs)
    stats = (
        combined.groupby(["Version", "Number of References", "Split_Value"])
        .describe()
        .reset_index()
    )
    stats.columns = [
        "Version",
        "Number of References",
        "Split_Value",
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]
    return stats
