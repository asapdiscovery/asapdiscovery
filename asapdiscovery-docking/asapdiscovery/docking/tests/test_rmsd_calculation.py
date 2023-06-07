from pathlib import Path

import numpy as np


def test_rmsd_calculation():
    """
    This function contains two unit tests that check the correctness of the calculate_rmsd_openeye function.

    The first test checks whether the RMSD between a reference molecule and itself is zero.
    It does this by loading the reference molecule from an SDF file using the load_openeye_sdf function from the
    asapdiscovery.data.openeye module and passing it twice to the calculate_rmsd_openeye function.
    It then uses the assert statement to check that the calculated RMSD is equal to zero.

    The second test checks whether the RMSD between a reference molecule and a query molecule is correct.
    It does this by loading the reference molecule and the query molecule from SDF files using the load_openeye_sdf
    function and passing them to the calculate_rmsd_openeye function.
    It then uses the assert statement to check that the calculated RMSD is equal to the pre-determined value.
    """
    from asapdiscovery.data.openeye import load_openeye_sdf
    from asapdiscovery.docking.analysis import calculate_rmsd_openeye

    input_dir = Path("inputs")

    ref_mol = load_openeye_sdf(str(input_dir / "Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf"))
    query_mol = load_openeye_sdf(
        str(input_dir / "ERI-UCB-ce40166b-17_Mpro-P2201_0A.sdf")
    )

    rmsd = calculate_rmsd_openeye(ref_mol, ref_mol)
    assert rmsd == 0.0

    rmsd = calculate_rmsd_openeye(ref_mol, query_mol)
    assert rmsd == 5.791467472680422


def test_writing_rmsd_calculation(tmp_path):
    """
    This function tests the ability to write all RMSD values between a reference molecule and a list of query molecules
    to a NumPy array file. It first loads a reference molecule and a query molecule from SDF files using the
    load_openeye_sdf function from the asapdiscovery.data.openeye module. It then calls the write_all_rmsds_to_reference
    function from the asapdiscovery.docking.analysis module to calculate and write RMSD values to a NumPy array file.
    The function finally loads the NumPy array file and compares to the pre-calculated reference for verification.
    """
    from asapdiscovery.data.openeye import load_openeye_sdf
    from asapdiscovery.docking.analysis import write_all_rmsds_to_reference

    input_dir = Path("inputs")

    ref_mol = load_openeye_sdf(str(input_dir / "Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf"))
    query_mol = load_openeye_sdf(
        str(input_dir / "ERI-UCB-ce40166b-17_Mpro-P2201_0A.sdf")
    )

    write_all_rmsds_to_reference(
        ref_mol, [query_mol, ref_mol, query_mol], tmp_path, "ERI-UCB-ce40166b-17"
    )
    rmsds = np.load(str(tmp_path / "ERI-UCB-ce40166b-17.npy"))

    reference = np.array(
        [
            ["ERI-UCB-ce40166b-17_Mpro-P2201_0A_0", "5.791467472680422"],
            ["Mpro-P0008_0A_ERI-UCB-ce40166b-17", "0.0"],
            ["ERI-UCB-ce40166b-17_Mpro-P2201_0A_0", "5.791467472680422"],
        ]
    )
    assert np.alltrue(reference == rmsds)
