import itertools
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union  # noqa: F401

import pandas as pd
import pkg_resources
import yaml


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

    @classmethod
    def is_in_values(cls, tag: str) -> bool:
        return tag in cls.get_values()

    @classmethod
    def from_iterable(cls, name: str, iter: Iterable) -> Enum:
        """
        Create a new Enum class from a set of tags
        """
        enum_data = {tag: tag for tag in iter}
        return cls(name, enum_data)


def make_target_tags(yaml_path: Union[str, Path]) -> tuple[Enum, set]:
    """
    Create a dynamic enum from a yaml file
    This enum contains all the target tags that are used in the manifold data
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
    viruses = data["virus"]
    target_tags = set()
    for v in viruses:
        for target in viruses[v]:
            target_tags.add(v + "_" + target)

    return TagEnumBase.from_iterable("TargetTags", target_tags), target_tags


def make_output_tags(yaml_path: Union[str, Path]) -> tuple[Enum, set]:
    """
    Create a dynamic enum from a yaml file
    This enum contains all the output tags that are used in the manifold data
    for example Docking_Score_POSIT

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

    return TagEnumBase.from_iterable("OutputTags", outputs), outputs


def make_static_tags(yaml_path) -> tuple[Enum, set]:
    """
    Create a dynamic enum from a yaml file
    This enum contains all the static tags that are used in the manifold data
    for example SMILES = SMILES

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
    static_identifiers = data["static_identifiers"]
    static_tags = set()
    for identifier in static_identifiers:
        static_tags.add(identifier)
    return TagEnumBase.from_iterable("StaticAndLegacyTags", static_tags), static_tags


# OK finally we can actually make the enums

# static path to the spec
manifold_data_spec = pkg_resources.resource_filename(
    __name__, "manifold_data_tags.yaml"
)

# make target enum and set
TargetTags, target_tag_set = make_target_tags(manifold_data_spec)

# make Output enum and set
OutputTags, output_tag_set = make_output_tags(manifold_data_spec)

# make static and legacy enum and set
StaticTags, static_tag_set = make_static_tags(manifold_data_spec)


def make_tag_combinations_and_combine_with_static(
    target_tags: set, output_tags: set, static_tags: set
) -> tuple[Enum, set]:
    """
    Make all possible combinations of target_tags and output_tags
    then add in the static and legacy tags
    """
    combos = set(itertools.product(output_tags, target_tags))
    combos = {combo[0] + "_" + combo[1] for combo in combos}
    final_tags = combos.union(static_tags)
    # sort the tags so that they are in alphabetical order
    final_tags = sorted(final_tags)
    return TagEnumBase.from_iterable("ManifoldAllowedTags", final_tags), final_tags


ManifoldAllowedTags, _ = make_tag_combinations_and_combine_with_static(
    target_tag_set, output_tag_set, static_tag_set
)


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
        return column in ManifoldAllowedTags.get_values()

    @staticmethod
    def all_valid_columns(columns: list[str], allow: Optional[list[str]] = []) -> bool:
        """
        Check if all columns are valid columns for the P5 comp-chem team to update
        """
        return all(
            [
                ManifoldFilter.is_allowed_column(column)
                for column in columns
                if column not in allow
            ]
        )

    @staticmethod
    def filter_dataframe_cols(
        df: pd.DataFrame, smiles_field=None, id_field=None, additional_cols=None
    ) -> pd.DataFrame:
        # construct list of allowed columns
        allowed_columns = ManifoldAllowedTags.get_values()
        if smiles_field is not None:
            allowed_columns.append(smiles_field)
        if id_field is not None:
            allowed_columns.append(id_field)
        if additional_cols is not None:
            allowed_columns.extend(additional_cols)

        # drop columns that are not allowed
        extra_cols = [col for col in df.columns if col not in allowed_columns]
        return df.drop(columns=extra_cols)
