from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from asapdiscovery.data.schema.ligand import Ligand

def check_ligand_series_uniqueness_and_names(ligands: list["Ligand"]) -> None:
    """
    Check the ligands are unique and have names.

    Args:
        ligands: The ligands to check for uniqueness and names.

    Raises:
        ValueError: If the ligands are not unique or have no names.
    """
    if len(set(ligands)) != len(ligands):
        count = Counter(ligands)
        duplicated = [key.compound_name for key, value in count.items() if value > 1]
        raise ValueError(
            f"ligand series contains {len(duplicated)} duplicate ligands: {duplicated}"
        )

    # if any ligands lack a name, then raise an exception; important for
    # ligands to have names for human-readable result gathering downstream
    if missing := len([ligand for ligand in ligands if not ligand.compound_name]):
        raise ValueError(
            f"{missing} of {len(ligands)} ligands do not have names; names are required for ligands for downstream results handling"
        )
