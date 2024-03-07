from asapdiscovery.data.backend.rdkit import load_sdf
import pytest


def test_ligand_sdf(moonshot_sdf, multipose_ligand, sdf_file):
    single_conf = load_sdf(moonshot_sdf)
    assert single_conf.GetNumConformers() == 1
    multiconf = load_sdf(multipose_ligand)

    assert multiconf.GetNumConformers() == 50

    # this should fail if the file has multiple ligands
    with pytest.raises(RuntimeError):
        load_sdf(sdf_file)
