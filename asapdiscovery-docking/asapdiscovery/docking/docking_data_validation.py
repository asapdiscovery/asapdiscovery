from enum import Enum
from typing import List, Optional  # noqa: F401


class DockingResultCols(str, Enum):
    DOCKING_CONFIDENCE_POSIT = "docking-confidence-POSIT"  # postera
    DOCKING_SCORE_POSIT = "docking-score-POSIT"  # postera
    DOCKING_STRUCTURE_POSIT = "docking-structure-POSIT"  # postera
    FITNESS_SCORE_FINT = "fitness-score-FINT"  # postera
    DOCKING_HIT = "docking-hit"  # postera
    SMILES = "SMILES"  # postera
    INCHIKEY = "INCHIKEY"  # postera
    COMPUTED_GAT_PIC50 = "computed-GAT-pIC50"  # postera
    COMPUTED_SCHNET_PIC50 = "computed-SchNet-pIC50"  # postera
    COMPUTED_E3NN_PIC50 = "computed-E3NN-pIC50"  # postera
    COMPUTED_GAT_LOGD = "computed-GAT-LogD"  # postera
    POSIT_METHOD = "_POSIT_method"
    LIGAND_ID = "ligand_id"
    TARGET_ID = "target_id"
    HTML_PATH_POSE = "html_path_pose"
    HTML_PATH_FITNESS = "html_path_fitness"
    GIF_PATH = "gif_path"
    MD_PATH_TRAJ = "md_path_traj"
    MD_PATH_MIN_PDB = "md_path_min_pdb"
    MD_PATH_FINAL_PDB = "md_path_final_pdb"
    SYMEXP_CLASHING = "symexp-clashing"  # postera
    SYMEXP_CLASH_NUM = "symexp-clash-num"  # postera

    @classmethod
    def get_columns(cls) -> list[str]:
        return [col.value for col in cls]
