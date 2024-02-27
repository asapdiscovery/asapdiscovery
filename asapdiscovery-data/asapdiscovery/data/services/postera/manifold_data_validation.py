import itertools
import logging
import warnings
from collections import defaultdict
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union  # noqa: F401

import pandas as pd
import pkg_resources
import yaml
from asapdiscovery.data.util.stringenum import StringEnum

logger = logging.getLogger(__name__)


# util function to open a yaml file and return the data
def load_yaml(yaml_path: Union[str, Path]) -> dict:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data


# we define a new Enum class with some handy methods


class TagEnumBase(StringEnum):
    @classmethod
    def is_in_values(cls, tag: str) -> bool:
        vals = cls.get_values()
        return tag in vals

    @classmethod
    def all_in_values(cls, query: list[str], allow: list[str] = []) -> bool:
        return all([cls.is_in_values(q) for q in query if q not in allow])

    @classmethod
    def from_iterable(cls, name: str, iter: Iterable) -> Enum:
        """
        Create a new Enum class from a set of tags
        """
        enum_data = {tag: tag for tag in iter}
        return cls(name, enum_data)

    @classmethod
    def filter_dataframe_cols(
        cls, df: pd.DataFrame, allow: Optional[list[str]] = None
    ) -> pd.DataFrame:
        # construct list of allowed columns
        allowed_columns = cls.get_values()

        if allow is not None:
            allowed_columns.extend(allow)

        # drop columns that are not allowed
        extra_cols = [col for col in df.columns if col not in allowed_columns]
        if len(extra_cols) > 0:
            warnings.warn(
                f"Columns {extra_cols} are not allowed. Dropping them from the dataframe"
            )
            logger.warn(
                f"Columns {extra_cols} are not allowed. Dropping them from the dataframe"
            )
        return df.drop(columns=extra_cols)

    @classmethod
    def get_values_underscored(cls):
        return [e.value.replace("-", "_") for e in cls]

    def get_value_underscored(self):
        return self.value.replace("-", "_")


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
    target_virus_map = {}
    virus_target_map = defaultdict(list)
    target_protein_map = {}
    for v in viruses:
        for target in viruses[v]:
            tag = v + "-" + target
            target_tags.add(tag)
            target_virus_map[tag] = v
            virus_target_map[v].append(target)
            target_protein_map[tag] = target

    return (
        TagEnumBase.from_iterable("TargetTags", target_tags),
        target_tags,
        target_virus_map,
        virus_target_map,
        target_protein_map,
    )


def make_virus_tags(yaml_path: Union[str, Path]) -> Enum:
    data = load_yaml(yaml_path)
    viruses = data["virus"]
    virus_tags = set()
    for v in viruses:
        virus_tags.add(v)
    return TagEnumBase.from_iterable("VirusTags", virus_tags)


def make_output_tags(yaml_path: Union[str, Path]) -> tuple[Enum, set, dict]:
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
    Set[str]
        Set of all the tags
    Dict[str, Tuple[str, str]]
        Dict linking the output name and the prefix and postfix
    """
    data = load_yaml(yaml_path)
    manifold_outputs = data["manifold_outputs"]
    outputs = set()

    prefix_postfix_dict = {}
    for output in manifold_outputs:
        key = list(output.keys())
        if len(key) > 1:
            raise ValueError("output should only have one key")
        output_name = key[0]

        try:
            units = output[output_name]["units"]
        except KeyError:
            raise ValueError("output should have a units key, even if it is empty")
        try:
            tools = output[output_name]["tools"]
        except KeyError:
            raise ValueError("output should have a tools key, even if it is empty")

        try:
            prefix = output[output_name]["prefix"]
        except KeyError:
            raise ValueError("output must have a prefix")

        try:
            postfix = output[output_name]["postfix"]
        except KeyError:
            raise ValueError("output must have a postfix")

        if tools:
            for tool in output[output_name]["tools"]:
                name = output_name + "-" + tool

                if units:
                    name += "-" + units
                outputs.add(name)
                prefix_postfix_dict[name] = (prefix, postfix)

        else:
            name = output_name
            if units:
                name += "-" + units

            outputs.add(name)
            prefix_postfix_dict[name] = (prefix, postfix)

    return (
        TagEnumBase.from_iterable("OutputTags", outputs),
        outputs,
        prefix_postfix_dict,
    )


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
(
    TargetTags,
    target_tag_set,
    TargetVirusMap,
    VirusTargetMap,
    TargetProteinMap,
) = make_target_tags(manifold_data_spec)

VirusTags = make_virus_tags(manifold_data_spec)

# make Output enum and set
OutputTags, output_tag_set, MANIFOLD_PREFIX_POSTFIX_DICT = make_output_tags(
    manifold_data_spec
)

# make static and legacy enum and set
StaticTags, static_tag_set = make_static_tags(manifold_data_spec)


def make_manifold_tag_name_from_components(
    pref: str, target: str, product: str, postf: str
) -> str:
    """
    Make a tag name from the components

    Parameters
    ----------
    pref : str
        Prefix
    target : str
        Target
    product : str
        Product
    postf : str
        Postfix
    """
    return pref + "_" + target + "_" + product + "_" + postf


def make_tag_combinations_and_combine_with_static(
    target_tags: set, output_tags: set, static_tags: set, prefix_postfix_dict: dict
) -> tuple[Enum, set]:
    """
    Make all possible combinations of target_tags and output_tags
    then add in the static and legacy tags
    """
    combos = set(itertools.product(output_tags, target_tags))

    combined = set()
    for combo in combos:
        product, target = combo
        pref, postf = prefix_postfix_dict[product]
        name = make_manifold_tag_name_from_components(pref, target, product, postf)
        combined.add(name)

    final_tags = combined.union(static_tags)
    # sort the tags so that they are in alphabetical order
    final_tags = sorted(final_tags)
    return TagEnumBase.from_iterable("ManifoldAllowedTags", final_tags), final_tags


ManifoldAllowedTags, _ = make_tag_combinations_and_combine_with_static(
    target_tag_set, output_tag_set, static_tag_set, MANIFOLD_PREFIX_POSTFIX_DICT
)


def map_output_col_to_manifold_tag(output_tags: Enum, target: str) -> dict[str, str]:
    """
    Build Postera tags given output tags and target. Only valid output tags in the enum
    are mapped to Postera tags.

    Parameters
    ----------
    output_tags : Enum
        Enum of output tags to produce Postera tags for.

    Returns
    -------
    mapping
        Output tags as keys, Postera tags as values.

    """
    mapping = {}
    for col in output_tags:
        if col.value in OutputTags.get_values():
            pref, post = MANIFOLD_PREFIX_POSTFIX_DICT[col.value]
            mapping[col.value] = make_manifold_tag_name_from_components(
                pref, target, col.value, post
            )
    return mapping


def drop_non_output_columns(
    df: pd.DataFrame, allow: Optional[list[str]] = []
) -> pd.DataFrame:
    """
    Drop columns of a docking result dataframe that are not allowed OutputTags
    ie the members of OutputTags.get_values() and StaticTags.get_values()

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe of docking results
    allow : list[str], optional
        List of additional columns to allow

    Returns
    -------
    df : pd.DataFrame
        Pandas dataframe with invalid columns dropped
    """
    output_cols = OutputTags.get_values()
    static_cols = StaticTags.get_values()
    # combine output and static columns
    output_cols.extend(static_cols)
    # add allowed columns
    output_cols.extend(allow)

    # drop all columns that are not in the output
    df = df.drop(columns=[col for col in df.columns if col not in output_cols])
    return df


def rename_output_columns_for_manifold(
    df: pd.DataFrame,
    target: str,
    output_enums: list[Enum],
    manifold_validate: Optional[bool] = True,
    drop_non_output: Optional[bool] = True,
    allow: Optional[list[str]] = [],
) -> pd.DataFrame:
    """
    Rename columns of a result dataframe that are available to be
    updated in the Postera Manifold for a specific target. i.e inject the
    target name into the column name to satisfy validation for Postera Manifold.
    for example:

    Also optionally drop columns of a result dataframe that are not allowed output tags
    ie the members of OutputTags.get_values() and StaticTags.get_values()

    docking-pose-POSIT -> in-silico_SARS-CoV-2-Mpro_docking-pose-POSIT_msk


    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe of docking results
    target : str
        Target name
    output_enums : list[Enum]
        List of enums to rename the columns of
    manifold_validate : bool, optional
        If True, validate that the columns are valid for Postera Manifold
    drop_non_output : bool, optional
        If True, drop columns that are not allowed output tags
    allow : list[str], optional
        List of additional columns to allow when dropping
    Returns
    -------
    df : pd.DataFrame
    Pandas dataframe with invalid columns dropped and valid columns renamed

    """
    if not TargetTags.is_in_values(target):
        raise ValueError(
            f"Target {target} is not valid. Valid targets are: {TargetTags.get_values()}"
        )

    if drop_non_output:
        df = drop_non_output_columns(df, allow=allow)

    mapping = {}
    for col_enum in output_enums:
        mapping.update(map_output_col_to_manifold_tag(col_enum, target))

    if manifold_validate:
        if not ManifoldAllowedTags.all_in_values(mapping.values()):
            raise ValueError(
                f"Columns in dataframe {mapping.values()} are not all valid for updating in postera. Valid columns are: {ManifoldAllowedTags.get_values()}"
            )
    # rename columns
    df = df.rename(columns=mapping)

    return df
