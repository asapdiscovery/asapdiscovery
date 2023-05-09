from asapdiscovery.data.schema import ExperimentalCompoundData
from asapdiscovery.data.schema import Ligand
import pytest

def test_classes():
    compound = ExperimentalCompoundData(compound_id="Test")
    assert compound.compound_id == "Test"

def test_ligand():
    l = Ligand(smiles="CC", id="XX112233", source="test", vc_id_postera="XX112233")

def test_ligand_invalid_smiles():
    with pytest.raises(ValueError):
        l = Ligand(smiles="GG", id="XX112233", source="test", vc_id_postera="XX112233")

def test_ligand_from_sdf(target_files):
    sdf, _, _ = target_files
    l = Ligand.from_sdf(sdf)
    assert l.smiles == "CC"