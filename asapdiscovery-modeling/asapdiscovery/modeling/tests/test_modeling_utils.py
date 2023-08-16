# This test suite can be run with a local path to save the output, ie:
# pytest test_modeling_utils.py --local_path=/path/to/save/files
# without a local path, output files will not be written

import pytest
from asapdiscovery.data.openeye import oechem
from asapdiscovery.modeling.modeling import split_openeye_mol
from asapdiscovery.modeling.schema import MoleculeFilter


# The main use cases for the modeling utils are:
# Getting just the protein
# Getting just the ligand in the active site
# Getting the protein and ligand in the active site
# Getting the protein and ligand and water
@pytest.mark.parametrize("target", ["sars"])
def test_simple_splitting(target, oemol_dict):
    oemol = oemol_dict[target]
    split_mol = split_openeye_mol(oemol)
    comp_dict = {
        "prot": (None, {"A", "B"}),
        "lig": (None, {"A", "B"}),
        "wat": ({"HOH"}, {"W"}),
    }
    for molecular_component, (res_name, chains) in comp_dict.items():
        comp_mol = split_mol[molecular_component]
        # Get all chains/res names to compare with correct results. Store all in sets
        #  for easy exact comparisons
        if res_name:
            assert {res.GetName() for res in oechem.OEGetResidues(comp_mol)} == res_name
        assert {res.GetChainID() for res in oechem.OEGetResidues(comp_mol)} == chains


@pytest.mark.parametrize(
    ("target", "ligand_chain"),
    [("sars", "A"), ("sars", "B"), ("mers", "B"), ("mers", "C")],
)
def test_ligand_splitting(target, local_path, ligand_chain, oemol_dict):
    """
    Test splitting when we just care about ligand.
    """

    oemol = oemol_dict[target]
    molfilter = MoleculeFilter(ligand_chain=ligand_chain)
    split_dict = split_openeye_mol(oemol, molfilter)

    # Make sure ligand only has ligand in it and only the right chain
    lig_res = oechem.OEGetResidues(split_dict["lig"])
    lig_chains = {res.GetChainID() for res in lig_res}
    assert all([res.IsHetAtom() and (not res.GetName() == "HOH") for res in lig_res])
    assert lig_chains == set(ligand_chain)


@pytest.mark.parametrize("protein_chains", [["A"], ["B"], ["A", "B"]])
@pytest.mark.parametrize("target", ["sars", "mers"])
def test_protein_splitting(target, local_path, protein_chains, oemol_dict):
    """
    Test splitting when we just care about protein.
    """

    oemol = oemol_dict[target]
    molfilter = MoleculeFilter(protein_chains=protein_chains)
    split_dict = split_openeye_mol(oemol, molfilter)

    # Make sure protein only has prot in it and only the right chain(s)
    prot_res = oechem.OEGetResidues(split_dict["prot"])
    prot_chains = {res.GetChainID() for res in prot_res}
    assert all([oechem.OEIsStandardProteinResidue(res) for res in prot_res])
    assert prot_chains == set(protein_chains)


@pytest.mark.parametrize(
    ("target", "ligand_chain"),
    [("sars", "A"), ("sars", "B"), ("mers", "B"), ("mers", "C")],
)
@pytest.mark.parametrize("protein_chains", [["A"], ["B"], ["A", "B"]])
def test_prot_and_lig_splitting(
    target, local_path, protein_chains, ligand_chain, oemol_dict
):
    """
    Test splitting when we care about protein and ligand.
    """

    oemol = oemol_dict[target]
    molfilter = MoleculeFilter(protein_chains=protein_chains, ligand_chain=ligand_chain)
    split_dict = split_openeye_mol(oemol, molfilter)

    # Make sure ligand only has ligand in it and only the right chain
    lig_res = oechem.OEGetResidues(split_dict["lig"])
    lig_chains = {res.GetChainID() for res in lig_res}
    assert all([res.IsHetAtom() and (not res.GetName() == "HOH") for res in lig_res])
    assert lig_chains == set(ligand_chain)

    # Make sure protein only has prot in it and only the right chain(s)
    prot_res = oechem.OEGetResidues(split_dict["prot"])
    prot_chains = {res.GetChainID() for res in prot_res}
    assert all([oechem.OEIsStandardProteinResidue(res) for res in prot_res])
    assert prot_chains == set(protein_chains)
