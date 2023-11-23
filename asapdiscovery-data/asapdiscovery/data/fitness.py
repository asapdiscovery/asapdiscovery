import json

import numpy as np
import pandas as pd
from asapdiscovery.data.metadata.resources import (
    SARS_CoV_2_fitness_data,
    ZIKV_NS2B_NS3pro_fitness_data,
    targets_with_fitness_data,
)
from asapdiscovery.data.postera.manifold_data_validation import TargetTags

_TARGET_TO_GENE = {
    TargetTags("SARS-CoV-2-Mpro").value: "nsp5 (Mpro)",
    TargetTags("MERS-CoV-Mpro").value: "not_available",
    TargetTags("SARS-CoV-2-Mac1").value: "nsp3",
}


def target_has_fitness_data(target: TargetTags) -> bool:
    return target in targets_with_fitness_data


def bloom_abstraction(fitness_scores_this_site: dict, threshold=-1.0) -> int:
    """
    Applies prescribed abstraction of how mutable a residue is given fitness data. Although the mean fitness
    was used at first, the current (2023.08.08) prescribed method is as follows (by Bloom et al):
    > something like “what is the number of mutations at a site that are reasonably well tolerated.” You could do this as something like number (or fraction) of mutations at a site that have a score >= -1 (that is probably a reasonable cutoff), using -1 as a cutoff where mutations start to cross from “highly deleterious” to “conceivably tolerated.”

    Parameters
    ----------
    fitness_scores_this_site: dict
        Dictionary containing fitness scores for a single site

    Returns
    -------
    num_tolerated_mutations: int

    """
    tolerated_mutations = [
        val for val in fitness_scores_this_site["fitness"] if val >= threshold
    ]
    return len(tolerated_mutations)


def apply_bloom_abstraction(fitness_dataframe: pd.DataFrame) -> dict:
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
    # add this column in case we're pulling in an experiment that has different data. We need to find
    # a good way of dealing with all this data coming from different labs.
    if not "expected_count" in fitness_dataframe.columns:
        fitness_dataframe["expected_count"] = 0

    fitness_dict = {}
    for idx, site_df in fitness_dataframe.groupby(by="site"):
        # remove wild type fitness score (this is always 0)
        fitness_scores_this_site = site_df[site_df["fitness"] != 0]

        # add all values to a dict
        fitness_dict[idx] = [
            bloom_abstraction(fitness_scores_this_site),
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


def normalize_fitness(fitness_df_abstract: pd.DataFrame) -> pd.DataFrame:
    """
    Read a pandas DF containing fitness data and normalizes values to 0-1. Normalization is as MinMax:
    - fitness: 0-100 ranges from non-fit to most fit (i.e., >>100 would mean residue is highly mutable).
    - confidence: 0-1 ranges from least confident to most confident.

    Parameters
    ----------
    fitness_df_abstract: pd.DataFrame
        Dataframe containing per-residue fitness data.

    Returns
    -------
    fitness_df_abstract: pd.DataFrame
        Dataframe containing per-residue fitness data normalized.
    """
    fitness_df_abstract["fitness"] = (
        (fitness_df_abstract["fitness"] - fitness_df_abstract["fitness"].min())
        / (fitness_df_abstract["fitness"].max() - fitness_df_abstract["fitness"].min())
        * 100
    )

    fitness_df_abstract["confidence"] = (
        fitness_df_abstract["confidence"] - fitness_df_abstract["confidence"].min()
    ) / (
        fitness_df_abstract["confidence"].max()
        - fitness_df_abstract["confidence"].min()
    )

    return fitness_df_abstract


def parse_fitness_json(target: TargetTags) -> pd.DataFrame:
    """
    Read a per-aa fitness JSON's specified target into a pandas DF.

    Parameters
    ----------
    target: str
        Specifies the target and virus, conforming to asapdiscovery.data.postera.manifold_data_validation.TargetTags

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

    if target not in ("SARS-CoV-2-Mpro", "SARS-CoV-2-Mac1", "ZIKV-NS2B-NS3pro"):
        raise NotImplementedError(
            f"Fitness data not yet available for {target}. Add to metadata if/when available."
        )

    ## START TODO
    if "SARS-CoV-2" in target:  # TODO: generalize this block
        with open(SARS_CoV_2_fitness_data) as f:
            data = json.load(f)
        data = data["data"]
        fitness_scores_bloom = pd.DataFrame(data)

        # now get the target-specific entries. Need to do because the phylo data is cross-genome.
        fitness_scores_bloom = fitness_scores_bloom[
            fitness_scores_bloom["gene"] == _TARGET_TO_GENE[target]
        ]

        if target == "SARS-CoV-2-Mac1":
            # need to subselect from nsp3 multidomain to get just Mac1. See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7113668/
            fitness_scores_bloom = fitness_scores_bloom[
                fitness_scores_bloom["site"].between(209, 372)
            ]
            fitness_scores_bloom[
                "site"
            ] -= 208  # reindex to set residue numbers to correct values
        elif target == "SARS-CoV-2-Mpro":
            # simpler; can just query the correct gene in the JSON.
            fitness_scores_bloom = fitness_scores_bloom[
                fitness_scores_bloom["gene"] == "nsp5 (Mpro)"
            ]
    # for ZIKV NS2B3 it's simpler - no need to subselect the target from the genome.
    elif target == "ZIKV-NS2B-NS3pro":
        with open(ZIKV_NS2B_NS3pro_fitness_data) as f:
            data = json.load(f)
        data = data["data"]
        fitness_scores_bloom = pd.DataFrame(data)
    ## END TODO

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
    # can instead return DF if ever we need to provide more info (top/worst mutation, confidence etc).
    fitness_df_abstract = normalize_fitness(fitness_df_abstract)
    return dict(zip(fitness_df_abstract.index, fitness_df_abstract["fitness"]))
