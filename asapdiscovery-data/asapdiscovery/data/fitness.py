import pandas as pd
import pkg_resources

from asapdiscovery.data.postera.manifold_data_validation import (
    TargetTags
    )

_TARGET_TO_GENE = {
    "SARS-CoV-2-Mpro" : "nsp5 (Mpro)",
    "MERS-CoV-Mpro" : "poop",
    "SARS-CoV-2-Mac1" : "poop",
                   } 

def parse_fitness_json(path, target) -> pd.DataFrame:
    """
    Read a per-aa fitness JSON's specified target into a pandas DF.

    Parameters
    ----------
    path : str
        Path to JSON file
    target: str
        Specifies the target and virus, conforming to dataviz.viz_targets


    Returns
    -------
    ... : pandas DataFrame
        poop
    """
    if target not in TargetTags.get_values():
        raise ValueError(f"Specified target is not valid, must be one of: {TargetTags.get_values()}")
    
    # load JSON by Bloom et al. JSONs for other virus genomes will be loaded here in the future. 
    fitness_json = pkg_resources.resource_filename(__name__, "../../../metadata/aa_fitness_sars_cov_2.json")

    
