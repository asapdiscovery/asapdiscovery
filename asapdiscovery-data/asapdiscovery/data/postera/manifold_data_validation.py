import pandas as pd
from enum import Enum


class ManifoldAllowedColumns(Enum):
    """
    Enum of allowed columns for the P5 comp-chem team to update in postera.
    """

    SMILES = "smiles"
    ID = "id"
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
