import json

import numpy as np
import pandas as pd
from asapdiscovery.data.metadata.resources import (
    SARS_CoV_2_fitness_data,
    ZIKV_NS2B_NS3pro_fitness_data,
    ZIKV_RdRppro_fitness_data,
    targets_with_fitness_data,
)
from asapdiscovery.data.services.postera.manifold_data_validation import (
    TargetTags,
    TargetVirusMap,
    VirusTags,
)

_TARGET_TO_GENE = {  # contains some entries for finding targets when subselecting a genome-wide fitness result.
    TargetTags("SARS-CoV-2-Mpro").value: "nsp5 (Mpro)",
    TargetTags("SARS-CoV-2-Mac1").value: "nsp3",
    TargetTags("SARS-CoV-2-N-protein").value: "N",
}

_TARGET_TO_FITNESS_DATA = {  # points to the vendored fitness data.
    TargetTags("SARS-CoV-2-Mpro").value: SARS_CoV_2_fitness_data,
    TargetTags("SARS-CoV-2-Mac1").value: SARS_CoV_2_fitness_data,
    TargetTags("SARS-CoV-2-N-protein").value: SARS_CoV_2_fitness_data,
    TargetTags("ZIKV-NS2B-NS3pro").value: ZIKV_NS2B_NS3pro_fitness_data,
    TargetTags("ZIKV-RdRppro").value: ZIKV_RdRppro_fitness_data,
}

_FITNESS_DATA_IS_CROSSGENOME = {  # sets whether the fitness data we have for this virus is the whole genome or a single target.
    VirusTags("SARS-CoV-2").value: True,
    VirusTags("ZIKV").value: False,
}

_FITNESS_DATA_FIT_THRESHOLD = {  # sets threshold at which a mutant is considered 'fit' for the specific fitness experiment. Directed by Bloom et al.
    VirusTags("SARS-CoV-2").value: -1.0,
    VirusTags("ZIKV").value: -1.0,  # this is OK for both NS2B-NS3pro and RdRppro
}


def target_has_fitness_data(target: TargetTags) -> bool:
    return target in targets_with_fitness_data


def bloom_abstraction(fitness_scores_this_site: dict, threshold: float) -> int:
    """
    Applies prescribed abstraction of how mutable a residue is given fitness data. Although the mean fitness
    was used at first, the current (2023.08.08) prescribed method is as follows (by Bloom et al):
    > something like “what is the number of mutations at a site that are reasonably well tolerated.” You could do this as something like number (or fraction) of mutations at a site that have a score >= -1 (that is probably a reasonable cutoff), using -1 as a cutoff where mutations start to cross from “highly deleterious” to “conceivably tolerated.”

    Parameters
    ----------
    fitness_scores_this_site: dict
        Dictionary containing fitness scores for a single site
    threshold: float
        fitness value to use as minimum value threshold to treat a mutation as acceptably fit.
    Returns
    -------
    num_tolerated_mutations: int

    """
    tolerated_mutations = [
        val for val in fitness_scores_this_site["fitness"] if val >= threshold
    ]
    return len(tolerated_mutations)


def apply_bloom_abstraction(fitness_dataframe: pd.DataFrame, threshold: float) -> dict:
    """
    Read a pandas DF containing fitness data parsed from a JSON in .parse_fitness_json() and return
    a processed dictionary with averaged fitness scores per residue. This is the current recommended
    method to get to a single value per residue. This function can be extended when the recommendation
    changes.

    Parameters
    ----------
    fitness_dataframe: pd.DataFrame
        DataFrame containing columns [gene, site, mutant, fitness, expected_count, wildtype]
    threshold: float
        fitness value to use as minimum value threshold to treat a mutation as acceptably fit.
    Returns
    -------
    fitness_dict : dict
        Dictionary where keys are residue indices underscored with chain IDs, keys are: [
            mean_fitness,
            wildtype_residue,
            most fit mutation,
            least fit mutation,
            total count (~confidence)
        ]
    """
    # add this column in case we're pulling in an experiment that has different data. We need to find
    # a good way of dealing with all this data coming from different labs. See Issue #649
    if "expected_count" not in fitness_dataframe.columns:
        fitness_dataframe["expected_count"] = 0

    fitness_dict = {}
    for (idx, chain), site_df in fitness_dataframe.groupby(by=["site", "chain"]):
        # remove wild type fitness score (this is always 0)
        fitness_scores_this_site = site_df[site_df["fitness"] != 0]

        # add all values to a dict
        fitness_dict[f"{idx}_{chain}"] = [
            bloom_abstraction(fitness_scores_this_site, threshold),
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
    return fitness_df_abstract

    # can reactivate below as required - return the above makes fitness categorical from 1 to n, where
    # n = number of fit mutants
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

    if not target_has_fitness_data(target):
        raise NotImplementedError(
            f"Fitness data not yet available for {target}. Add to metadata if/when available."
        )

    fitness_scores_bloom = get_fitness_scores_bloom_by_target(target)

    threshold = _FITNESS_DATA_FIT_THRESHOLD[TargetVirusMap[target]]

    # now apply the abstraction currently recommended by Bloom et al to get to a single float per residue.
    fitness_dict_abstract = apply_bloom_abstraction(fitness_scores_bloom, threshold)

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


def get_fitness_scores_bloom_by_target(target: TargetTags) -> pd.DataFrame:
    # find the virus that corresponds to the target
    virus = TargetVirusMap[target]
    # find the fitness data that corresponds to the virus
    fitness_data = _TARGET_TO_FITNESS_DATA[target]
    # read the fitness data into a dataframe
    with open(fitness_data) as f:
        data = json.load(f)
    if "data" in data.keys():
        # this is SARS-CoV-2 cross-genome phylo data - can directly grab 'data' key from json.
        data = data["data"]
        fitness_scores_bloom = pd.DataFrame(data)
    elif "ZIKV NS2B-NS3 (Closed)" in data.keys():
        # this is ZIKV NS2B-NS3 DMS data - need to grab data differently from json.
        data = data["ZIKV NS2B-NS3 (Closed)"]["mut_metric_df"]
        fitness_scores_bloom = pd.DataFrame(data).rename(
            columns={"reference_site": "site", "Log2(Effect)": "fitness"}
        )

    if _FITNESS_DATA_IS_CROSSGENOME[virus]:
        # now get the target-specific entries. Need to do because the phylo data is cross-genome.
        fitness_scores_bloom = fitness_scores_bloom[
            fitness_scores_bloom["gene"] == _TARGET_TO_GENE[target]
        ]
    else:
        pass  # no need to subselect

    # post-processing for specific targets
    # TODO: replace all the below by a more intelligent + robust alignment algorithm.
    if target == "SARS-CoV-2-Mac1":
        # need to subselect from nsp3 multidomain to get just Mac1. See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7113668/
        fitness_scores_bloom = fitness_scores_bloom[
            fitness_scores_bloom["site"].between(209, 372)
        ]
        fitness_scores_bloom["site"] -= 204  # PDB starts at resindex 5
        fitness_scores_bloom["chain"] = "A"

    elif target == "SARS-CoV-2-Mpro":
        fitness_scores_bloom["chain"] = "A"
    elif target == "SARS-CoV-2-N-protein":
        # For N-protein, we want to show both monomers in the dimer because they are inter-locked and ligands may bind the interface,
        # so can't get away with showing one of the monomers as blue. We'll make separate rows in the data for each chain.
        # first double the DF.
        doubled_df = pd.concat([fitness_scores_bloom] * 2).reset_index()

        # now add chain A/C to the first/second half of the doubled DF.
        doubled_df.loc[: len(fitness_scores_bloom), "chain"] = "A"
        doubled_df.loc[len(fitness_scores_bloom) :, "chain"] = "C"
        if not len(doubled_df[doubled_df["chain"] == "A"].values) == len(
            doubled_df[doubled_df["chain"] == "C"].values
        ):
            raise ValueError(
                "Chain lengths between chains A/C are not equal - unable to naively duplicate fitness data across; please debug."
            )
        else:
            fitness_scores_bloom = doubled_df
    elif target == "ZIKV-NS2B-NS3pro":
        # cursed. TODO: replace this with an auto-align.
        ns2b_section = fitness_scores_bloom[
            fitness_scores_bloom["site"].str.contains("NS2B")
        ]
        ns2b_section.loc[ns2b_section.index, "site"] = [
            int(site.replace("(NS2B)", "")) for site in ns2b_section["site"].values
        ]
        ns2b_section = ns2b_section[ns2b_section["site"].between(46, 89)]

        # tag NS2B as chain A
        ns2b_section["chain"] = "A"

        # repeat for NS3
        ns3_section = fitness_scores_bloom[
            fitness_scores_bloom["site"].str.contains("NS3")
        ]
        ns3_section.loc[ns3_section.index, "site"] = [
            int(site.replace("(NS3)", "")) for site in ns3_section["site"].values
        ]
        ns3_section = ns3_section[ns3_section["site"].between(10, 177)]

        # tag NS3 as chain B
        ns3_section["chain"] = "B"

        # then add back together and treat as normal downstream.
        fitness_scores_bloom = pd.concat([ns2b_section, ns3_section])

    return fitness_scores_bloom
