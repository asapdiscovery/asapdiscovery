from enum import Enum
from typing import List  # noqa: F401

from asapdiscovery.data.postera.manifold_data_validation import (
    ManifoldAllowedTags,
    TargetTags,
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
    Columns for docking results that are target dependent. That is they can have
    the target name appended to them. e.g 

    Docking_Confidence_POSIT + sars2_Mpro = Docking_Confidence_POSIT_sars2_Mpro

    These cannot be reordered as the order is important for the output list.
    (See #358)

    Columns that are not in the allowed Postera Manifold columns are prefixed with an underscore
    """

    _DU_STRUCTURE = "_du_structure"
    _DOCKED_FILE = "_docked_file"
    _POSE_ID = "_pose_id"
    _DOCKED_RMSD = "_docked_RMSD"
    Docking_Confidence_POSIT = "Docking_Confidence_POSIT"
    _POSIT_METHOD = "_POSIT_method"
    Docking_Score_POSIT = "Docking_Score_POSIT"
    _CLASH = "_clash"
    ML_Score_GAT_pIC50 = "ML_Score_GAT_pIC50"
    ML_Score_Schnet_pIC50 = "ML_Score_Schnet_pIC50"

    @classmethod
    def get_columns(cls) -> list[str]:
        return [col.value for col in cls]

    @staticmethod
    def get_columns_for_target(target: str) -> list[str]:
        if not TargetTags.is_in_values(target):
            raise ValueError(
                f"Target {target} is not valid. Valid targets are: {TargetTags.get_values()}"
            )
        return [col.value + f"_{target}" for col in TargetDependentCols]

    @staticmethod
    def get_columns_for_target_with_manifold_validation(target: str) -> list[str]:
        cols = [
            col.value + f"_{target}"
            for col in TargetDependentCols
            if col.value in ManifoldAllowedTags.get_values()
        ]
        if not ManifoldFilter.all_valid_columns(cols):
            raise ValueError(
                f"Columns in dataframe {cols} are not all valid for updating in postera. Valid columns are: {ManifoldAllowedColumns.get_columns()}"
            )
        return cols
