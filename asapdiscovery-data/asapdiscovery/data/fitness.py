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
    for idx, site_df in fitness_scores_bloom.groupby(by="site"):
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


def parse_fitness_json(target) -> pd.DataFrame:
    """
    Read a per-aa fitness JSON's specified target into a pandas DF.

    Parameters
    ----------
    target: str
        Specifies the target and virus, conforming to dataviz.viz_targets

    Returns
    -------
    fitness_scores : pandas DataFrame
        Dataframe where indices are residue numbers,  `fitness` column contains fitness score
    """
    if target not in TargetTags.get_values():
        raise ValueError(
            f"Specified target is not valid, must be one of: {TargetTags.get_values()}"
        )

    # load JSON by Bloom et al. JSONs for other virus genomes will be loaded here in the future.
    fitness_json = pkg_resources.resource_filename(
        __name__, "../../../metadata/aa_fitness_sars_cov_2.json"
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

    print(fitness_dict)
