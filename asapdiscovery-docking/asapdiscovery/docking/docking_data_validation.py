from enum import Enum
from typing import List, Optional  # noqa: F401

import pandas as pd


class DockingResultCols(Enum):
    """
    Columns for docking results, i.e. the output of the docking pipeline
    These cannot be reordered as the order is important for the output list.
    (See #358)

    Columns that are not in the allowed Postera Manifold columns are prefixed with an underscore
    """

    LIGAND_ID = "ligand_id"
    DU_STRUCTURE = "_du_structure"
    DOCKED_FILE = "_docked_file"
    POSE_ID = "_pose_id"
    DOCKED_RMSD = "_docked_RMSD"
    DOCKING_CONFIDENCE_POSIT = "docking-confidence-POSIT"  # postera
    POSIT_METHOD = "_POSIT_method"
    DOCKING_SCORE_POSIT = "docking-score-POSIT"  # postera
    CLASH = "_clash"
    SMILES = "SMILES"  # postera
    COMPUTED_GAT_PIC50 = "computed-GAT-pIC50"  # postera
    COMPUTED_SCHNET_PIC50 = "computed-SchNet-pIC50"  # postera

    @classmethod
    def get_columns(cls) -> list[str]:
        return [col.value for col in cls]
