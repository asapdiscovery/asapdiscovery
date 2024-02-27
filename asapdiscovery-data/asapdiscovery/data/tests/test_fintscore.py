from pathlib import Path

from asapdiscovery.data.backend.openeye import load_openeye_pdb
from asapdiscovery.data.backend.plip import compute_fint_score
from asapdiscovery.data.readers.molfile import MolFileFactory
from asapdiscovery.data.testing.test_resources import fetch_test_file


def test_fint_score():
    fint_score = compute_fint_score(
        load_openeye_pdb(
            Path(
                fetch_test_file(
                    "Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.pdb"
                )
            )
        ),
        MolFileFactory(
            filename=Path(fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf"))
        )
        .load()[0]
        .to_oemol(),
        "SARS-CoV-2-Mpro",
    )
    # should return a tuple
    assert isinstance(fint_score, tuple)

    # both should be floats
    assert isinstance(fint_score[0], float)
    assert isinstance(fint_score[1], float)

    # should both fall between 0 and 1
    assert 0 <= fint_score[0] <= 1.0
    assert 0 <= fint_score[1] <= 1.0
