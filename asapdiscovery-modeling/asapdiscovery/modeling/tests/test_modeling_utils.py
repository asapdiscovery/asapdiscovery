from collections import namedtuple
from pathlib import Path

import pytest
from asapdiscovery.data.openeye import (
    load_openeye_cif1,
    load_openeye_pdb,
    oechem,
    save_openeye_pdb,
)
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.modeling.modeling import find_component_chains, split_openeye_mol
from asapdiscovery.modeling.schema import MoleculeFilter


@pytest.fixture
def sars_pdb():
    return fetch_test_file("Mpro-P2660_0A_bound.pdb")


@pytest.fixture
def mers_pdb():
    return fetch_test_file("rcsb_8czv-assembly1.cif")


@pytest.fixture
def sars_oe(sars_pdb):
    # Load structure
    prot = load_openeye_pdb(str(sars_pdb))
    assert type(prot) == oechem.OEGraphMol
    return prot


@pytest.fixture
def mers_oe(mers_pdb):
    # Load structure
    prot = load_openeye_cif1(str(mers_pdb))
    assert type(prot) == oechem.OEGraphMol
    return prot


@pytest.fixture
def oemol_dict(sars_oe, mers_oe):
    return {"sars": sars_oe, "mers": mers_oe}


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
    for molecular_component in ["protein", "ligand"]:
        if molecular_component in components:
            assert find_component_chains(split_mol, molecular_component) == ["A", "B"]
        else:
            assert find_component_chains(split_mol, molecular_component) == []


@pytest.mark.parametrize(
    "components",
    [["ligand"], ["protein", "ligand"], ["protein", "ligand", "water"]],
)
@pytest.mark.parametrize(
    ("target", "ligand_chain"),
    [("sars", "A"), ("sars", "B"), ("mers", "B"), ("mers", "C")],
)
def test_pdb_ligand_splitting(
    target, local_path, files, ligand_chain, components, oemol_dict
):
    oemol = oemol_dict[target]
    molfilter = MoleculeFilter(
        components_to_keep=components,
        ligand_chain=ligand_chain,
    )
    complex = split_openeye_mol(oemol, molfilter)

    save_openeye_pdb(
        complex,
        Path(
            local_path,
            f"split_test_{target}_{'-'.join(components)}_lig{ligand_chain}.pdb",
        ),
    )
    assert find_component_chains(complex, "ligand") == [ligand_chain]


@pytest.mark.parametrize(
    "components",
    [["protein"], ["protein", "ligand"], ["protein", "ligand", "water"]],
)
@pytest.mark.parametrize("protein_chains", [["A"], ["B"], ["A", "B"]])
@pytest.mark.parametrize("target", ["sars", "mers"])
def test_pdb_protein_splitting(
    target, local_path, files, protein_chains, components, oemol_dict
):
    oemol = oemol_dict[target]
    molfilter = MoleculeFilter(
        components_to_keep=components,
        protein_chains=protein_chains,
    )
    complex = split_openeye_mol(oemol, molfilter)
    assert find_component_chains(complex, "protein") == protein_chains

    save_openeye_pdb(
        complex,
        Path(
            local_path,
            f"split_test_{target}_{'-'.join(components)}_prot{''.join(protein_chains)}.pdb",
        ),
    )
