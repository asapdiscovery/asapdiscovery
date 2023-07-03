from enum import Enum
from pathlib import Path
from typing import Union, Tuple
import pandas as pd
import yaml
import itertools
import pkg_resources


# util function to open a yaml file and return the data
def load_yaml(yaml_path: Union[str, Path]) -> dict:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data


# we define a new Enum class with some handy methods


class TagEnumBase(Enum):
    @classmethod
    def get_values(cls):
        return [e.value for e in cls]

    @classmethod
    def get_names(cls):
        return [e.name for e in cls]


def make_bio_tags(yaml_path: Union[str, Path]) -> Tuple[Enum, set]:
    """
    Create a dynamic enum from a yaml file
    This enum contains all the biology tags that are used in the manifold data
    for example sars2_Mpro = sars2_Mpro

    Parameters
    ----------
    yaml_path : Union[str, Path]
        Path to the yaml file containing the tags

    Returns
    -------
    Enum
        Enum containing all the tags
    set of str
        Set of all the tags
    """
    data = load_yaml(yaml_path)
    organisms = data["organism"]
    bio_tags = set()
    for org in organisms:
        for target in organisms[org]:
            bio_tags.add(org + "_" + target)
    # make the same tags also the values
    enum_data = {tag: tag for tag in bio_tags}
    return TagEnumBase("BioTags", enum_data), bio_tags


def make_output_tags(yaml_path: Union[str, Path]) -> Tuple[Enum, set]:
    data = load_yaml(yaml_path)
    manifold_outputs = data["manifold_outputs"]
    outputs = set()
    for output in manifold_outputs:
        key = list(output.keys())
        if len(key) > 1:
            raise ValueError("output should only have one key")
        output_name = key[0]

        try:
            has_units = output[output_name]["units"]
        except KeyError:
            raise ValueError("output should have a units key, even if it is empty")
        try:
            has_tools = output[output_name]["tools"]
        except KeyError:
            raise ValueError("output should have a tools key, even if it is empty")

        if has_tools:
            for tool in output[output_name]["tools"]:
                name = output_name + "_" + tool
                if has_units:
                    name += "_" + output[output_name]["units"]
                outputs.add(name)
        else:
            name = output_name
            if has_units:
                name += "_" + output[output_name]["units"]
            outputs.add(name)
    enum_data = {tag: tag for tag in outputs}
    return TagEnumBase("OutputTags", enum_data), outputs


manifold_data_spec = pkg_resources.resource_filename(
    __name__, "manifold_data_tags.yaml"
)

BioTags, biotag_set = make_bio_tags(manifold_data_spec)

OutputTags, outputtag_set = make_output_tags(manifold_data_spec)


class ManifoldAllowedColumns(Enum):
    """
    Enum of allowed columns for the P5 comp-chem team to update in postera.
    """

    SMILES = "SMILES"
    LIGAND_ID = "ligand_id"
    ASAP_VC_ID = "asap_vc_id"
    MERS_POSE = "mers_pose"
    SARS2_POSE = "sars2_pose"
    POSIT_PROB_MERS = "POSIT_prob_mers"
    POSIT_PROB_SARS2 = "POSIT_prob_sars2"
    CHEMGAUSS4_SCORE_MERS = "chemgauss4_score_mers"
    CHEMGAUSS4_SCORE_SARS2 = "chemgauss4_score_sars2"
    DOCKED_FILE_MERS = "docked_file_mers"
    DOCKED_FILE_SARS2 = "docked_file_sars2"
    GAT_SCORE_MERS = "GAT_score_mers"
    GAT_SCORE_SARS2 = "GAT_score_sars2"
    SCHNET_SCORE_MERS = "SCHNET_score_mers"
    SCHNET_SCORE_SARS2 = "SCHNET_score_sars2"

    def get_columns():
        return [column.value for column in ManifoldAllowedColumns]


class ManifoldFilter:
    """
    Class to filter columns and data to only those that the P5 comp-chem team
    should be able to update in postera.
    """

    @staticmethod
    def is_allowed_column(column: str) -> bool:
        """
        Check if a column is a valid column for the P5 comp-chem team to update
        """
        return column in ManifoldAllowedColumns.get_columns()

    @staticmethod
    def all_valid_columns(columns: list[str]) -> bool:
        """
        Check if all columns are valid columns for the P5 comp-chem team to update
        """
        return all([ManifoldFilter.is_allowed_column(column) for column in columns])

    @staticmethod
    def filter_dataframe_cols(
        df: pd.DataFrame, smiles_field=None, id_field=None, additional_cols=None
    ) -> pd.DataFrame:
        # construct list of allowed columns
        allowed_columns = ManifoldAllowedColumns.get_columns()
        if smiles_field is not None:
            allowed_columns.append(smiles_field)
        if id_field is not None:
            allowed_columns.append(id_field)
        if additional_cols is not None:
            allowed_columns.extend(additional_cols)

        # drop columns that are not allowed
        extra_cols = [col for col in df.columns if col not in allowed_columns]
        return df.drop(columns=extra_cols)
