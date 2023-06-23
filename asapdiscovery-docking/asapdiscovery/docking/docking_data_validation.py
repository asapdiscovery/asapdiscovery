from enum import Enum
from typing import List

from asapdiscovery.data.postera.manifold_data_validation import ManifoldAllowedColumns


class DockingResultCols(Enum):
    """
    Columns for docking results
    """

    LIGAND_ID = "ligand_id"
    DU_STRUCTURE = "du_structure"
    DOCKED_FILE = "docked_file"
    POSE_ID = "pose_id"
    DOCKED_RMSD = "docked_RMSD"
    POSIT_PROB = "POSIT_prob"
    POSIT_METHOD = "POSIT_method"
    CHEMGAUSS4_SCORE = "chemgauss4_score"
    CLASH = "clash"
    SMILES = "SMILES"
    GAT_SCORE = "GAT_score"
    SCHNET_SCORE = "SCHNET_score"

    @staticmethod
    def get_columns() -> list[str]:
        return [col.value for col in DockingResultCols]


class TargetDependentCols(Enum):
    """
    Columns that are target dependent
    """

    DU_STRUCTURE = "du_structure"
    DOCKED_FILE = "docked_file"
    POSE_ID = "pose_id"
    DOCKED_RMSD = "docked_RMSD"
    POSIT_PROB = "POSIT_prob"
    POSIT_METHOD = "POSIT_method"
    CHEMGAUSS4_SCORE = "chemgauss4_score"
    GAT_SCORE = "GAT_score"
    SCHNET_SCORE = "SCHNET_score"

    @staticmethod
    def get_columns() -> list[str]:
        return [col.value for col in TargetDependentCols]

    @staticmethod
    def get_columns_for_target(target: str) -> list[str]:
        return [col.value + f"_{target}" for col in TargetDependentCols]

    @staticmethod
    def get_columns_for_target_with_manifold_validation(target: str) -> list[str]:
        cols = [col.value + f"_{target}" for col in TargetDependentCols]
        if not ManifoldFilter.all_valid_columns(cols):
            raise ValueError(
                f"Columns in dataframe {cols} are not all valid for updating in postera. Valid columns are: {ManifoldAllowedColumns.get_columns()}"
            )
        return cols
