# This test suite can be run with a local path to save the output, ie:
# pytest test_modeling_utils.py --local_path=/path/to/save/files
# without a local path, output files will not be written

from pathlib import Path

import pytest
from asapdiscovery.data.openeye import save_openeye_pdb
from asapdiscovery.modeling.modeling import find_component_chains, split_openeye_mol
from asapdiscovery.modeling.schema import MoleculeFilter


# The main use cases for the modeling utils are:
# Getting just the protein
# Getting just the ligand in the active site
# Getting the protein and ligand in the active site
# Getting the protein and ligand and water
@pytest.mark.parametrize(
    "components",
    ["ligand", "protein", ["ligand", "protein"], ["ligand", "protein", "water"]],
)
@pytest.mark.parametrize("target", ["sars"])
def test_simple_splitting(target, components, oemol_dict):
    oemol = oemol_dict[target]
    split_mol = split_openeye_mol(oemol, components)
    for molecular_component in ["protein", "ligand", "water"]:
        res_name, chains = (
            ("HOH", ["W"]) if molecular_component == "water" else (None, ["A", "B"])
        )

        if molecular_component in components:
            assert (
                find_component_chains(split_mol, molecular_component, res_name)
                == chains
            )
        else:
            assert find_component_chains(split_mol, molecular_component, res_name) == []


@pytest.mark.parametrize(
    "components",
    [["ligand"], ["protein", "ligand"], ["protein", "ligand", "water"]],
)
@pytest.mark.parametrize(
    ("target", "ligand_chain"),
    [("sars", "A"), ("sars", "B"), ("mers", "B"), ("mers", "C")],
)
def test_ligand_splitting(target, local_path, ligand_chain, components, oemol_dict):
    oemol = oemol_dict[target]
    molfilter = MoleculeFilter(
        components_to_keep=components,
        ligand_chain=ligand_chain,
    )
    complex = split_openeye_mol(oemol, molfilter)

    if local_path:
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
def test_protein_splitting(target, local_path, protein_chains, components, oemol_dict):
    oemol = oemol_dict[target]
    molfilter = MoleculeFilter(
        components_to_keep=components,
        protein_chains=protein_chains,
    )
    complex = split_openeye_mol(oemol, molfilter)
    if local_path:
        save_openeye_pdb(
            complex,
            Path(
                local_path,
                f"split_test_{target}_{'-'.join(components)}_prot{''.join(protein_chains)}.pdb",
            ),
        )
    assert find_component_chains(complex, "protein") == protein_chains
