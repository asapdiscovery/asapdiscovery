import pytest
from asapdiscovery.data.schema.ligand import Ligand

from asapdiscovery.alchemy.alchemize import (
    compute_clusters,
    calc_mcs_residuals,
    rescue_outsiders,
    write_clusters,
)

from rdkit import Chem


# def compute_clusters(asap_ligands, outsider_number, console):


# def partial_sanitize(mol):


def test_calc_mcs_residuals():
    assert calc_mcs_residuals(  # simple check
        Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CCCCC")
    ) == (0, 0)

    assert calc_mcs_residuals(  # more complex check, good core overlap
        Chem.MolFromSmiles(
            "CC(C)(c1nc2c(C(=O)Nc3ccc4c(c3)CNC4)cc(Cl)cc2[nH]1)S(C)(=O)=O"
        ),
        Chem.MolFromSmiles(
            "CN1C(=O)CC[C@H]1c1nc2c(C(=O)Nc3ccc4c(c3)CNC4)cc(Cl)cc2[nH]1"
        ),
    ) == (7, 7)

    assert calc_mcs_residuals(  # more complex check, bad core overlap
        Chem.MolFromSmiles(
            "CC(C)(c1nc2c(C(=O)Nc3ccc4c(c3)CNC4)cc(Cl)cc2[nH]1)S(C)(=O)=O"
        ),
        Chem.MolFromSmiles("FC(F)(F)c1nc(no1)c3cc(c(OCCCc2onc(c2)C)c(c3)C)C"),
    ) == (22, 19)


# def rescue_outsiders(
#     outsiders, alchemical_clusters, max_transform, processors, console
# ):


# def write_clusters(alchemical_clusters, clusterfiles_prefix, outsiders):
#     """Stores clusters to individual SDF files using the clusterfiles prefix variable"""
