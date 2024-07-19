import tempfile
from glob import glob
import pytest
from asapdiscovery.alchemy.alchemize import (
    calc_mcs_residuals,
    compute_clusters,
    partial_sanitize,
    rescue_outsiders,
    write_clusters,
)
from asapdiscovery.data.schema.ligand import Ligand, write_ligands_to_multi_sdf
from rdkit import Chem
from asapdiscovery.alchemy.cli.cli import alchemy
from click.testing import CliRunner


@pytest.fixture() 
def test_ligands():
        
    TEST_LIGANDS = [
        Ligand.from_smiles(smi, compound_name="foo")
        for smi in [
            "O=C(NC1=CC(Cl)=CC(C(=O)NC2=CC=C(CC3CCNCC3)C=C2)=C1)OCC1=CC=CC=C1",
            "CCNC(=O)NC1=CC(Cl)=CC(C(=O)NC2=CC(C)=CC(CN)=C2)=C1",
            "NC1=CC=C(NC(=O)C2=CC(Cl)=CC3=C2C=NN3)C=N1",
            "NCC1=CC=CC(NC(=O)C2=CC(Cl)=CC(CN)=C2)=C1",
            "O=C(C1=CC=CC2=CC=CC=C12)NC3=CC=C4CNCC4=C3",
            "CCNC(=O)NC1=CC(Cl)=CC(C(=O)NC2=CC(C)=CC(CN)=C2)=C1",
            "O=C(C1=CC=CC2=C(F)C=CC=C12)NC3=CC=C4CNCC4=C3",
            "O=C(C1=CC=CC2=C(Cl)C=CC=C12)NC3=CC=C4CNCC4=C3",
            "O=C(C1=CC=CC2=C(Br)C=CC=C12)NC3=CC=C4CNCC4=C3",
        ]
    ]
    return TEST_LIGANDS


@pytest.fixture()
def test_ligands_sdfile(test_ligands, tmp_path):
    # write the ligands to a temporary SDF file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sdf", delete=False, dir=tmp_path
    ) as f:
        write_ligands_to_multi_sdf(f.name, test_ligands, overwrite=True)
    return f.name




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


def test_cli(test_ligands_sdfile, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        alchemy,
        [
            "prep",
            "alchemize",
            "-l",
            test_ligands_sdfile,
            "-n",
            "tst",
            "-onu",
            "2",
            "-mt", 
            "9",

        ],
    )
    assert result.exit_code == 0
