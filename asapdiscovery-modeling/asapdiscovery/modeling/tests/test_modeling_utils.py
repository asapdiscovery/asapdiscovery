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
    find_ligand_chains,
    find_protein_chains,
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


# The main use cases for the modeling utils are:
# Getting just the protein
# Getting just the ligand in the active site
# Getting the protein and ligand in the active site
# Getting the protein and ligand and water
@pytest.mark.parametrize("components", ["ligand", "protein", ["ligand", "protein"]])
def test_simple_splitting(sars_oe, local_path, components):
    split_mol = split_openeye_mol(sars_oe, components)
    if "ligand" in components:
        assert find_ligand_chains(split_mol) == ["A", "B"]
    else:
        assert find_ligand_chains(split_mol) == []
    if "protein" in components:
        assert find_protein_chains(split_mol) == ["A", "B"]
    else:
        assert find_protein_chains(split_mol) == []


@pytest.mark.parametrize("ligand_chain", ["A", "B"])
@pytest.mark.parametrize(
    "components",
    [["ligand"], ["protein", "ligand"], ["protein", "ligand", "water"]],
)
def test_pdb_ligand_splitting(sars_oe, local_path, files, ligand_chain, components):
    molfilter = MoleculeFilter(
        components_to_keep=components,
        ligand_chain=ligand_chain,
    )
    complex = split_openeye_mol(sars_oe, molfilter)
    assert find_ligand_chains(complex) == [ligand_chain]

    save_openeye_pdb(
        complex,
        Path(
            local_path,
            f"split_test_{'-'.join(components)}_lig{ligand_chain}.pdb",
        ),
    )


@pytest.mark.parametrize(
    "components",
    [["protein"], ["protein", "ligand"], ["protein", "ligand", "water"]],
)
@pytest.mark.parametrize("protein_chains", [["A"], ["B"], ["A", "B"]])
def test_pdb_protein_splitting(sars_oe, local_path, files, protein_chains, components):
    molfilter = MoleculeFilter(
        components_to_keep=components,
        protein_chains=protein_chains,
    )
    complex = split_openeye_mol(sars_oe, molfilter)
    assert find_protein_chains(complex) == protein_chains

    save_openeye_pdb(
        complex,
        Path(
            local_path,
            f"split_test_{'-'.join(components)}_prot{''.join(protein_chains)}.pdb",
        ),
    )
