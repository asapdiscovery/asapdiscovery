import pandas as pd
import rich


def print_header(console: "rich.Console"):
    """Print an ASAP-Alchemy header"""

    console.line()
    console.rule("ASAP-Alchemy")
    console.line()


def pull_from_postera(molecule_set_name: str):
    """
    A convenience method with tucked imports to avoid importing Postera tools when not needed.

    Args:
        The name of the molecule set which should be pulled from postera

    Returns:
        A list of Ligands extracted from postera molecule set.
    """
    from asapdiscovery.data.services.postera.postera_factory import PosteraFactory
    from asapdiscovery.data.services.services_config import PosteraSettings

    # this will pull the settings from environment variables
    settings = PosteraSettings()
    return PosteraFactory(settings=settings, molecule_set_name=molecule_set_name).pull()


def upload_to_postera(molecule_set_name: str, target: str, absolute_dg_predictions: pd.DataFrame) -> bool:
    """
    A convenience method to format predicted absolute DG values using Alchemy and upload to postera with tucked imports
    to avoid importing Postera tools.

    Args:
        molecule_set_name: The name of the molecule set in postera the results should be attached to.
        target: The name of the biological target this result is associated with.
        absolute_dg_predictions: The dataset of absolute dg predictions created by asap-alchemy.

    Returns:
        `True` if a new molecule set was made else `False` if we just update an existing set.
    """
    from asapdiscovery.data.services.postera.postera_uploader import PosteraUploader
    from asapdiscovery.data.services.services_config import PosteraSettings
    from asapdiscovery.alchemy.predict import dg_to_postera_dataframe
    from asapdiscovery.data.services.postera.manifold_data_validation import rename_output_columns_for_manifold
    from enum import Enum

    # mock some enum to specifiy which columns are allowed?
    class AlchemyResults(str, Enum):
        SMILES = "SMILES"
        INCHI_KEY = "Inchi_Key"
        LIGAND_ID = "Ligand_ID"
        COMPUTED_BIOCHEMICAL_ACTIVITY_FEC = "computed-biochemical-activity-FEC"
        COMPUTED_BIOCHEMICAL_ACTIVITY_FEC_UNCERTAINTY = "computed-biochemical-activity-FEC-uncertainty"

    # convert the dg values to pIC50 with the expected names
    postera_df = dg_to_postera_dataframe(absolute_predictions=absolute_dg_predictions)
    result_df = rename_output_columns_for_manifold(
        df=postera_df,
        target=target,
        output_enums=[AlchemyResults],
        manifold_validate=True,
        drop_non_output=True,
        allow=[
            AlchemyResults.SMILES.value,
            AlchemyResults.INCHI_KEY.value,
            AlchemyResults.LIGAND_ID.value
        ],
    )

    postera_uploader = PosteraUploader(
        settings=PosteraSettings(),
        molecule_set_name=molecule_set_name,
        id_field=AlchemyResults.LIGAND_ID.value,
        smiles_field=AlchemyResults.SMILES.value
    )

    _, _, made_new_molset = postera_uploader.push(result_df)

    return made_new_molset
