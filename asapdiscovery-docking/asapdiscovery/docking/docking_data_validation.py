from enum import Enum
from typing import List  # noqa: F401

from asapdiscovery.data.postera.manifold_data_validation import (
    ManifoldAllowedColumns,
    ManifoldFilter,
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
