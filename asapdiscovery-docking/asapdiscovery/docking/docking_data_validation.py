from enum import Enum
from typing import List, Optional  # noqa: F401

import pandas as pd
from asapdiscovery.data.postera.manifold_data_validation import (
    ManifoldAllowedTags,
    ManifoldFilter,
    OutputTags,
    StaticTags,
    TargetTags,
)


class DockingResultCols(Enum):
    """
    Columns for docking results, i.e. the output of the docking pipeline
    These cannot be reordered as the order is important for the output list.
    (See #358)

    Columns that are not in the allowed Postera Manifold columns are prefixed with an underscore
    """

    _LIGAND_ID = "_ligand_id"
    _DU_STRUCTURE = "_du_structure"
    _DOCKED_FILE = "_docked_file"
    _POSE_ID = "_pose_id"
    _DOCKED_RMSD = "_docked_RMSD"
    Docking_Confidence_POSIT = "Docking_Confidence_POSIT"
    _POSIT_METHOD = "_POSIT_method"
    Docking_Score_POSIT = "Docking_Score_POSIT"
    _CLASH = "_clash"
    SMILES = "SMILES"
    ML_Score_GAT_pIC50 = "ML_Score_GAT_pIC50"
    ML_Score_Schnet_pIC50 = "ML_Score_Schnet_pIC50"

    @classmethod
    def get_columns(cls) -> list[str]:
        return [col.value for col in cls]

    @classmethod
    def get_columns_for_target(cls, target: str) -> list[str]:
        if not TargetTags.is_in_values(target):
            raise ValueError(
                f"Target {target} is not valid. Valid targets are: {TargetTags.get_values()}"
            )
        return [col.value + f"_{target}" for col in cls]

    @classmethod
    def get_columns_for_target_with_manifold_validation(cls, target: str) -> list[str]:
        if not TargetTags.is_in_values(target):
            raise ValueError(
                f"Target {target} is not valid. Valid targets are: {TargetTags.get_values()}"
            )

        cols = [
            col.value + f"_{target}"
            for col in cls
            if ManifoldFilter.is_allowed_column(col.value + f"_{target}")
        ]
        if not ManifoldFilter.all_valid_columns(cols):
            raise ValueError(
                f"Columns in dataframe {cols} are not all valid for updating in postera. Valid columns are: {ManifoldAllowedTags.get_values()}"
            )
        return cols


def drop_non_output_columns(
    df: pd.DataFrame, allow: Optional[list[str]] = []
) -> pd.DataFrame:
    """
    Drop columns of a docking result dataframe that are not allowed output tags
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


def rename_output_columns_for_target(
    df: pd.DataFrame, target: str, manifold_validate: Optional[bool] = True
) -> pd.DataFrame:
    """
    Rename columns of a docking result dataframe that are available to be 
    updated in the Postera Manifold for a specific target. i.e inject the 
    target name into the column name to satisfy validation for Postera Manifold. 
    for example:

    Docking_Score_POSIT -> Docking_Score_POSIT_sars2_mpro
    
    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe of docking results
    target : str
        Target name
    manifold_validate : bool, optional
        If True, validate that the columns are valid for Postera Manifold
    
    Returns
    -------
    df : pd.DataFrame
        Pandas dataframe with valid columns renamed
    """
    if not TargetTags.is_in_values(target):
        raise ValueError(
            f"Target {target} is not valid. Valid targets are: {TargetTags.get_values()}"
        )

    # make mapping between keys in DockingResultCols and the target specific columns
    mapping = {
        col.value: col.value + f"_{target}"
        for col in DockingResultCols
        if col.value in OutputTags.get_values()
    }

    if manifold_validate:
        if not ManifoldFilter.all_valid_columns(mapping.values()):
            raise ValueError(
                f"Columns in dataframe {mapping.values()} are not all valid for updating in postera. Valid columns are: {ManifoldAllowedTags.get_values()}"
            )

    # rename columns
    df = df.rename(columns=mapping)
    return df


def drop_and_rename_output_cols_for_target(
    df: pd.DataFrame,
    target: str,
    manifold_validate: Optional[bool] = True,
    allow: Optional[list[str]] = [],
) -> pd.DataFrame:
    """
    Drop columns of a docking result dataframe that are not allowed output tags
    ie the members of OutputTags.get_values() and StaticTags.get_values()
    and then rename columns of a docking result dataframe that are available to be 
    updated in the Postera Manifold for a specific target. i.e inject the 
    target name into the column name to satisfy validation for Postera Manifold. 
    for example:

    Docking_Score_POSIT -> Docking_Score_POSIT_sars2_mpro
    _blahblah -> None (dropped)

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe of docking results
    target : str
        Target name
    manifold_validate : bool, optional
        If True, validate that the columns are valid for Postera Manifold
    allow : list[str], optional
        List of additional columns to allow
    
    Returns
    -------
    df : pd.DataFrame
    Pandas dataframe with invalid columns dropped and valid columns renamed

    """
    df_dropped = drop_non_output_columns(df, allow=allow)
    df_dropped = rename_output_columns_for_target(
        df_dropped, target, manifold_validate=manifold_validate
    )
    return df_dropped
