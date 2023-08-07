import json

import numpy as np
import pandas as pd
import pkg_resources
from asapdiscovery.data.postera.manifold_data_validation import TargetTags

_TARGET_TO_GENE = {
    "SARS-CoV-2-Mpro": "nsp5 (Mpro)",
    "MERS-CoV-Mpro": "TBD",
    "SARS-CoV-2-Mac1": "TBD",
}


def apply_bloom_abstraction(fitness_dataframe) -> dict:
    """
    Read a pandas DF containing fitness data parsed from a JSON in .parse_fitness_json() and return
    a processed dictionary with averaged fitness scores per residue. This is the current recommended
    method to get to a single value per residue. This function can be extended when the recommendation
    changes.

    Parameters
    ----------
    fitness_dataframe: pd.DataFrame
        DataFrame containing columns [gene, site, mutant, fitness, expected_count, wildtype]

    Returns
    -------
    fitness_dict : dict
        Dictionary where keys are residue indices, keys are: [
            mean_fitness,
            wildtype_residue,
            most fit mutation,
            least fit mutation,
            total count (~confidence)
        ]
    """

    fitness_dict = {}
    for idx, site_df in fitness_dataframe.groupby(by="site"):
        # remove wild type fitness score (this is always 0)
        fitness_scores_this_site = site_df[site_df["fitness"] != 0]

        # add all values to a dict
        fitness_dict[idx] = [
            np.mean(fitness_scores_this_site["fitness"].values),  # compute mean fitness
            fitness_scores_this_site["wildtype"].values[0],  # wildtype residue
            fitness_scores_this_site.sort_values(by="fitness")["mutant"].values[
                -1
            ],  # most fit mutation
            fitness_scores_this_site.sort_values(by="fitness")["mutant"].values[
                0
            ],  # least fit mutation
            np.sum(fitness_scores_this_site["expected_count"].values),  # total count
        ]
    return fitness_dict


def normalize_fitness(fitness_df_abstract) -> pd.DataFrame:
    """
    Read a pandas DF containing fitness data and normalizes values to 0-1. Normalization is as MinMax:
    - fitness: 0-1 ranges from non-fit to most fit (i.e., >>1 would mean residue is highly mutable).
    - confidence: 0-1 ranges from least confident to most confident.

    Parameters
    ----------
    fitness_df_abstract: pd.DataFrame
        Dataframe containing per-residue fitness data.

    Returns
    -------
    fitness_df_abstract: pd.DataFrame
        Dataframe containing per-residue fitness data normalized to 0-1.
    """
    fitness_df_abstract["fitness"] = (
        fitness_df_abstract["fitness"] - fitness_df_abstract["fitness"].min()
    ) / (fitness_df_abstract["fitness"].max() - fitness_df_abstract["fitness"].min())

    fitness_df_abstract["confidence"] = (
        fitness_df_abstract["confidence"] - fitness_df_abstract["confidence"].min()
    ) / (
        fitness_df_abstract["confidence"].max()
        - fitness_df_abstract["confidence"].min()
    )

    return fitness_df_abstract


def parse_fitness_json(target) -> pd.DataFrame:
    """
    Read a per-aa fitness JSON's specified target into a pandas DF.

    Parameters
    ----------
    target: str
        Specifies the target and virus, conforming to dataviz.viz_targets

    Returns
    -------
    fitness_df_abstract : pandas DataFrame
        Dataframe where indices are residue numbers, columns are:
            "fitness" -> normalized fitness (0 is not mutable, 1 is highly mutable)
            "wildtype_residue" 
            "most_fit_mutation" 
            "least_fit_mutation" 
            "confidence" -> normalized confidence (0 is not confident, 1 is highly confident)
    """
    if target not in TargetTags.get_values():
        raise ValueError(
            f"Specified target is not valid, must be one of: {TargetTags.get_values()}"
        )

    # load JSON by Bloom et al. JSONs for other virus genomes will be loaded here in the future.
    fitness_json = pkg_resources.resource_filename(
        __name__,
        "../../../metadata/aa_fitness_sars_cov_2.json",  # TODO change when `metadata` is vendored properly
    )

    with open(fitness_json) as f:
        data = json.load(f)
    data = data["data"]
    fitness_scores_bloom = pd.DataFrame(data)

    # now get the target-specific entries.
    fitness_scores_bloom = fitness_scores_bloom[
        fitness_scores_bloom["gene"] == _TARGET_TO_GENE[target]
    ]

    # now apply the abstraction currently recommended by Bloom et al to get to a single float per residue.
    fitness_dict_abstract = apply_bloom_abstraction(fitness_scores_bloom)
    fitness_df_abstract = pd.DataFrame.from_dict(
        fitness_dict_abstract,
        orient="index",
        columns=[
            "fitness",
            "wildtype_residue",
            "most_fit_mutation",
            "least_fit_mutation",
            "confidence",
        ],
    )
    fitness_df_abstract.index.name = "residue"

    # normalize fitness and confidence values to 0-1 for easier parsing by visualizers downstream and return df.
    return normalize_fitness(fitness_df_abstract)
