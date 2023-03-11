from pathlib import Path

import numpy as np
from asapdiscovery.data.openeye import load_openeye_sdf
from asapdiscovery.docking.analysis import (
    calculate_rmsd_openeye,
    write_all_rmsds_to_reference,
)

output_dir = Path("outputs")
input_dir = Path("inputs")

ref_mol = load_openeye_sdf(str(input_dir / "Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf"))
query_mol = load_openeye_sdf(str(input_dir / "ERI-UCB-ce40166b-17_Mpro-P2201_0A.sdf"))


def test_rmsd_calculation():
    rmsd = calculate_rmsd_openeye(ref_mol, ref_mol)
    assert rmsd == 0.0

    rmsd = calculate_rmsd_openeye(ref_mol, query_mol)
    assert rmsd == 5.791467472680422


def test_writing_rmsd_calculation():
    write_all_rmsds_to_reference(
        ref_mol, [query_mol], output_dir, "ERI-UCB-ce40166b-17"
    )
    rmsds = np.load(output_dir / "ERI-UCB-ce40166b-17.npy")
    print(rmsds)
