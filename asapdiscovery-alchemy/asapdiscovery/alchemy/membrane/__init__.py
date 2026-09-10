"""Building membrane-embedded receptors for alchemical free energy calculations.

The output is the single, fully built, solvated, pre-equilibrated PDB that OpenFE's
membrane workflow consumes as a ``ProteinMembraneComponent``.

Typical use::

    from asapdiscovery.alchemy.membrane import build_membrane_system, get_preset

    build_membrane_system(
        receptor_pdb="prepared_receptor.pdb",
        spec=get_preset("herg"),
        output_prefix="8ZYO",
        equilibration_ns=20.0,
        pocket_ligand="reference_ligand.sdf",
    )

or from the command line, ``asap-alchemy membrane build``.
"""

from .builder import (
    align_receptor,
    build_membrane_system,
    composition,
    diagnostics,
    lipid_patch,
    wrap_molecules,
)
from .geometry import (
    best_inplane_angle,
    check_landmarks,
    membrane_axis,
    membrane_placement,
    read_pdb_atoms,
    rotation_to_z,
    structural_ion_positions,
)
from .schema import (
    HERG,
    PRESETS,
    EquilibrationStage,
    MembraneAlignmentSpec,
    MembraneBuildSettings,
    ResidueRange,
    StructuralIonSite,
    get_preset,
)

__all__ = [
    "HERG",
    "PRESETS",
    "EquilibrationStage",
    "MembraneAlignmentSpec",
    "MembraneBuildSettings",
    "ResidueRange",
    "StructuralIonSite",
    "align_receptor",
    "best_inplane_angle",
    "build_membrane_system",
    "check_landmarks",
    "composition",
    "diagnostics",
    "get_preset",
    "lipid_patch",
    "membrane_axis",
    "membrane_placement",
    "read_pdb_atoms",
    "rotation_to_z",
    "structural_ion_positions",
    "wrap_molecules",
]
