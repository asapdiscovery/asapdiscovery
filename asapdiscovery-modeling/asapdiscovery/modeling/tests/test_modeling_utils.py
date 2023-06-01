import pytest
from pathlib import Path
from collections import namedtuple
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.data.openeye import (
    load_openeye_pdb,
    save_openeye_pdb,
    oechem,
)
from asapdiscovery.modeling.modeling import (
    remove_extra_ligands,
    find_ligand_chains,
    split_openeye_mol,
)
from asapdiscovery.modeling.schema import MoleculeFilter


@pytest.fixture
def sars_pdb():
    return fetch_test_file("Mpro-P2660_0A_bound.pdb")


@pytest.fixture
def sars_oe(sars_pdb):
    # Load structure
    prot = load_openeye_pdb(str(sars_pdb))
    assert type(prot) == oechem.OEGraphMol
    return prot


@pytest.fixture
def files(tmp_path_factory, local_path):
    if not type(local_path) == str:
        outdir = tmp_path_factory.mktemp("test_prep")
    else:
        outdir = Path(local_path)
    files = namedtuple("files", ["ligA", "ligB"])
    paths = [outdir / f"{name}.pdb" for name in files._fields]
    return files(*paths)


def test_pdb_processors(sars_oe, local_path, files):
    # Test removing extra ligands
    assert find_ligand_chains(sars_oe) == ["A", "B"]
    assert find_ligand_chains(remove_extra_ligands(sars_oe, lig_chain="A")) == ["A"]
    assert find_ligand_chains(remove_extra_ligands(sars_oe, lig_chain="B")) == ["B"]

    # Test getting only the protein and ligand
    split_mol = split_openeye_mol(sars_oe)
    assert find_ligand_chains(split_mol["lig"]) == ["A"]
    assert find_ligand_chains(split_mol["other"]) == ["B"]

    molfilter = MoleculeFilter(components_to_keep=["protein", "ligand", "water"])
