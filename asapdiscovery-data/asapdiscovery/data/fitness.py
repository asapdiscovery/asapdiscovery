import json

import pandas as pd
import pkg_resources
from asapdiscovery.data.postera.manifold_data_validation import TargetTags

_TARGET_TO_GENE = {
    "SARS-CoV-2-Mpro": "nsp5 (Mpro)",
    "MERS-CoV-Mpro": "TBD",
    "SARS-CoV-2-Mac1": "TBD",
}


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
    fitness_scores = pd.DataFrame(data)
    print(fitness_scores)
