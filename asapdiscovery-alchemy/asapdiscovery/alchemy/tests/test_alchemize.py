from glob import glob

from asapdiscovery.alchemy.alchemize import (
    calc_mcs_residuals,
    compute_clusters,
    partial_sanitize,
    rescue_outsiders,
    write_clusters,
)
from rdkit import Chem


def test_compute_clusters(test_ligands):
    outsiders, alchemical_clusters = compute_clusters(test_ligands, outsider_number=2)

    # check that the clusters are what we would expect them to be
    assert outsiders, alchemical_clusters == 2
    assert list(outsiders.keys()) == [
        "c1ccc(COCNc2cccc(CNc3ccc(CC4CCNCC4)cc3)c2)cc1",
        "c1cncc(NCc2cccc3[nH]ncc23)c1",
    ]
    assert list(alchemical_clusters.keys()) == [
        "c1ccc(CNc2ccccc2)cc1",
        "c1ccc2c(CNc3ccc4c(c3)CNC4)cccc2c1",
    ]
    assert [ligand[0].smiles for ligand in outsiders.values()] == [
        "c1ccc(cc1)COC(=O)Nc2cc(cc(c2)Cl)C(=O)Nc3ccc(cc3)CC4CCNCC4",
        "c1cc(ncc1NC(=O)c2cc(cc3c2cn[nH]3)Cl)N",
    ]
    assert [ligand[0].smiles for ligand in alchemical_clusters.values()] == [
        "CCNC(=O)Nc1cc(cc(c1)Cl)C(=O)Nc2cc(cc(c2)CN)C",
        "c1ccc2c(c1)cccc2C(=O)Nc3ccc4c(c3)CNC4",
    ]


def test_partial_sanitize():
    # simple example should always sanitize:
    assert isinstance(partial_sanitize(Chem.MolFromSmiles("CC")), Chem.rdchem.Mol)

    # complex example (pleconaril) should always sanitize:
    assert isinstance(
        partial_sanitize(
            Chem.MolFromSmiles("FC(F)(F)c1nc(no1)c3cc(c(OCCCc2onc(c2)C)c(c3)C)C")
        ),
        Chem.rdchem.Mol,
    )

    # complex example (pleconaril) should always sanitize:
    assert isinstance(
        partial_sanitize(
            Chem.MolFromSmiles("FC(F)(F)c1nc(no1)c3cc(c(OCCCc2onc(c2)C)c(c3)C)C")
        ),
        Chem.rdchem.Mol,
    )

    # a molecule with multiple valency errors should also sanitize with this method:
    assert isinstance(
        partial_sanitize(
            Chem.MolFromSmiles(
                "FC(F)(c1onc(c2[W]c(C)c(C)c(C)c2(C)(C)C)n1)F", sanitize=False
            )
        ),
        Chem.rdchem.Mol,
    )


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


def test_rescue_outsiders(test_ligands):
    outsiders, alchemical_clusters = compute_clusters(test_ligands, outsider_number=2)

    # we already check the initial clusters in `test_compute_clusters()`
    assert outsiders, alchemical_clusters == 2

    # now rescue the outsiders (singletons)
    resc_outsiders, resc_alchemical_clusters = rescue_outsiders(
        outsiders, alchemical_clusters, max_transform=9, processors=1
    )

    # we should have rescued one outsider, one remains
    assert len(resc_outsiders) == 1
    assert len(resc_alchemical_clusters) == 2


def test_write_clusters(test_ligands, tmp_path):
    # generate clusters and write them to a tmp dir
    outsiders, alchemical_clusters = compute_clusters(test_ligands, outsider_number=2)
    clusterfiles_prefix = f"{tmp_path}/test_cluster"
    write_clusters(alchemical_clusters, clusterfiles_prefix, outsiders)

    written_files = glob(f"{tmp_path}/test_cluster*")

    # check that we have two alchemical cluster files written and one outsiders file
    assert len(written_files) == 3
    assert len([file for file in written_files if "outsiders" in file]) == 1
