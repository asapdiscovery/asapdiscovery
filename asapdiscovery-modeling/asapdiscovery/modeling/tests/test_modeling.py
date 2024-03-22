import pytest
from asapdiscovery.data.backend.openeye import load_openeye_pdb, oechem
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.modeling.modeling import split_openeye_mol
from asapdiscovery.modeling.schema import MoleculeFilter


@pytest.fixture()
def oemol():
    return load_openeye_pdb(fetch_test_file("Mpro-P2660_0A_bound.pdb"))


# The main use cases for the modeling utils are:
# Getting just the protein
# Getting just the ligand in the active site
# Getting the protein and ligand in the active site
# Getting the protein and ligand and water
def test_simple_splitting(oemol):
    split_mol = split_openeye_mol(oemol, keep_one_lig=False)
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


def test_simple_splitting_keep_one_lig(oemol):
    lig_mol = split_openeye_mol(oemol)["lig"]
    assert {res.GetChainID() for res in oechem.OEGetResidues(lig_mol)} == {"A"}


@pytest.mark.parametrize("ligand_chain", ["A", "B"])
def test_ligand_splitting(ligand_chain, oemol):
    """
    Test splitting when we just care about ligand.
    """

    molfilter = MoleculeFilter(ligand_chain=ligand_chain)
    split_dict = split_openeye_mol(oemol, molfilter)

    # Make sure ligand only has ligand in it and only the right chain
    lig_res = oechem.OEGetResidues(split_dict["lig"])
    lig_chains = {res.GetChainID() for res in lig_res}
    assert all([res.IsHetAtom() and (not res.GetName() == "HOH") for res in lig_res])
    assert lig_chains == set(ligand_chain)


@pytest.mark.parametrize("protein_chains", [["A"], ["B"], ["A", "B"]])
def test_protein_splitting(protein_chains, oemol):
    """
    Test splitting when we just care about protein.
    """

    molfilter = MoleculeFilter(protein_chains=protein_chains)
    split_dict = split_openeye_mol(oemol, molfilter)

    # Make sure protein only has prot in it and only the right chain(s)
    prot_res = oechem.OEGetResidues(split_dict["prot"])
    prot_chains = {res.GetChainID() for res in prot_res}
    assert all([oechem.OEIsStandardProteinResidue(res) for res in prot_res])
    assert prot_chains == set(protein_chains)


@pytest.mark.parametrize("ligand_chain", ["A", "B"])
@pytest.mark.parametrize("protein_chains", [["A"], ["B"], ["A", "B"]])
def test_prot_and_lig_splitting(protein_chains, ligand_chain, oemol):
    """
    Test splitting when we care about protein and ligand.
    """

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
