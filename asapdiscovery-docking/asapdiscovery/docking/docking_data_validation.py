from enum import Enum
from typing import List, Optional  # noqa: F401


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


# make a new enum for the postera manifold columns that actually exist
# so we can deprecate the other one in time.


class DockingResultColsV2(str, Enum):
    DOCKING_CONFIDENCE_POSIT = "docking-confidence-POSIT"  # postera
    DOCKING_SCORE_POSIT = "docking-score-POSIT"  # postera
    DOCKING_STRUCTURE_POSIT = "docking-structure-POSIT"  # postera
    DOCKING_HIT = "docking-hit"  # postera
    SMILES = "SMILES"  # postera
    COMPUTED_GAT_PIC50 = "computed-GAT-pIC50"  # postera
    COMPUTED_SCHNET_PIC50 = "computed-SchNet-pIC50"  # postera
    POSIT_METHOD = "_POSIT_method"
    LIGAND_ID = "ligand_id"
    TARGET_ID = "target_id"
    HTML_PATH_POSE = "html_path_pose"
    HTML_PATH_FITNESS = "html_path_fitness"
    GIF_PATH = "gif_path"
    MD_PATH_TRAJ = "md_path_traj"
    MD_PATH_MIN_PDB = "md_path_min_pdb"
    MD_PATH_FINAL_PDB = "md_path_final_pdb"

    @classmethod
    def get_columns(cls) -> list[str]:
        return [col.value for col in cls]
