import pathlib

import pytest
from asapdiscovery.alchemy.cli.cli import alchemy
from asapdiscovery.alchemy.schema.fec import (
    FreeEnergyCalculationFactory,
    FreeEnergyCalculationNetwork,
)
from asapdiscovery.alchemy.schema.prep_workflow import (
    AlchemyDataSet,
    AlchemyPrepWorkflow,
)
from asapdiscovery.data.testing.test_resources import fetch_test_file
from click.testing import CliRunner
from rdkit import Chem


def test_alchemy_create(tmpdir):
    """Test making a workflow file using the cli"""

    runner = CliRunner()

    with tmpdir.as_cwd():
        result = runner.invoke(
            alchemy,
            ["create", "workflow.json"],
        )
        assert result.exit_code == 0
        # make sure we can load the factory
        _ = FreeEnergyCalculationFactory.from_file("workflow.json")


def test_alchemy_plan_from_raw(tmpdir, tyk2_protein, tyk2_ligands):
    """Make sure we can plan networks using the CLI"""

    runner = CliRunner()

    with tmpdir.as_cwd():
        # write the files to the current folder
        tyk2_protein.to_pdb_file("tyk2_protein.pdb")
        # write the ligands to a single sdf file
        with Chem.SDWriter("tyk2_ligands.sdf") as output:
            for ligand in tyk2_ligands:
                output.write(ligand.to_rdkit())

        # call the cli
        result = runner.invoke(
            alchemy,
            [
                "plan",
                "-n",
                "tyk2-testing",
                "-l",
                "tyk2_ligands.sdf",
                "-r",
                "tyk2_protein.pdb",
            ],
        )
        assert result.exit_code == 0
        # try and open the planned network
        network = FreeEnergyCalculationNetwork.from_file(
            "tyk2-testing/planned_network.json"
        )
        # make sure all ligands are in the network
        assert len(network.network.ligands) == len(tyk2_ligands)


def test_alchemy_plan_from_alchemy_dataset(tmpdir):
    """Try and run the alchemy planning method from an alchemy dataset"""
    # grab a pre computed alchemy dataset
    alchemy_data = fetch_test_file("prepared_alchemy_dataset.json")

    runner = CliRunner()

    with tmpdir.as_cwd():
        result = runner.invoke(
            alchemy, ["plan", "-ad", alchemy_data.as_posix(), "-n", "mac1-testing"]
        )
        assert result.exit_code == 0
        assert "Loading Ligands and protein from AlchemyDataSet" in result.stdout
        # try and open the planned network
        network = FreeEnergyCalculationNetwork.from_file(
            "mac1-testing/planned_network.json"
        )
        # make sure all ligands are in the network
        assert len(network.network.ligands) == 3


def test_alchemy_plan_missing():
    """make sure an informative error is raised if we do not provide all the inputs required"""

    runner = CliRunner()

    with pytest.raises(RuntimeError) as error:
        _ = runner.invoke(
            alchemy, ["plan", "-n", "test-planner"], catch_exceptions=False
        )

    assert (
        "Please provide either an AlchemyDataSet created with `asap-alchemy prep run` or ligand and receptor input files."
        == str(error.value)
    )


def test_alchemy_prep_create(tmpdir):
    """Test creating the alchemy prep workflow"""

    runner = CliRunner()

    with tmpdir.as_cwd():
        # write a workflow with a core smarts and make sure it can be loaded
        result = runner.invoke(
            alchemy, ["prep", "create", "-f", "prep-workflow.json", "-cs", "CC"]
        )
        assert result.exit_code == 0
        prep_workflow = AlchemyPrepWorkflow.parse_file("prep-workflow.json")
        assert prep_workflow.core_smarts == "CC"


def test_alchemy_prep_run_with_fails(tmpdir, mac1_complex):
    """Test running the alchemy prep workflow on a set of mac1 ligands and that failures are captured"""

    # locate the ligands input file
    ligand_file = fetch_test_file("constrained_conformer/mac1_ligands.smi")

    runner = CliRunner()

    with tmpdir.as_cwd():
        # complex to a local file
        mac1_complex.to_json_file("complex.json")

        result = runner.invoke(
            alchemy,
            [
                "prep",
                "run",
                "-n",
                "mac1-testing",
                "-l",
                ligand_file.as_posix(),
                "-r",
                "complex.json",
            ],
        )
        assert result.exit_code == 0
        # make sure stereo enum is run
        assert (
            "[✓] StereoExpander successful,  number of unique ligands 5."
            in result.stdout
        )
        # make sure stereo is detected
        assert (
            "! WARNING the reference structure is chiral, check output structures carefully!"
            in result.stdout
        )
        # check all molecules have poses made
        assert "[✓] Pose generation successful for 5/5." in result.stdout
        # 2 molecules should be removed due to inconsistent stereo
        assert (
            "[✓] Stereochemistry filtering complete 2 molecules removed."
            in result.stdout
        )
        # check a warning is printed if some molecules are removed
        assert (
            "WARNING some ligands failed to have poses generated see failed_ligands"
            in result.stdout
        )
        # check we can load the result
        prep_dataset = AlchemyDataSet.from_file(
            "mac1-testing/prepared_alchemy_dataset.json"
        )
        # make sure the receptor is writen to file
        assert (
            pathlib.Path(prep_dataset.dataset_name)
            .joinpath(f"{prep_dataset.reference_complex.target.target_name}.pdb")
            .exists()
        )
        assert prep_dataset.dataset_name == "mac1-testing"
        assert len(prep_dataset.input_ligands) == 5
        assert len(prep_dataset.posed_ligands) == 3
        assert len(prep_dataset.failed_ligands["InconsistentStereo"]) == 2


def test_alchemy_prep_run_all_pass(tmpdir, mac1_complex):
    """Test running the alchemy prep workflow and make sure all ligands pass when expected."""

    # locate the ligands input file
    ligand_file = fetch_test_file("constrained_conformer/mac1_ligands.smi")

    runner = CliRunner()

    with tmpdir.as_cwd():
        # complex to a local file
        mac1_complex.to_json_file("complex.json")
        # create a new prep workflow which allows incorrect stereo
        workflow = AlchemyPrepWorkflow(strict_stereo=False)
        workflow.to_file("workflow.json")

        result = runner.invoke(
            alchemy,
            [
                "prep",
                "run",
                "-f",
                "workflow.json",
                "-n",
                "mac1-testing",
                "-l",
                ligand_file.as_posix(),
                "-r",
                "complex.json",
            ],
        )
        assert result.exit_code == 0
        # make sure stereo enum is run
        assert (
            "[✓] StereoExpander successful,  number of unique ligands 5."
            in result.stdout
        )
        # make sure stereo is detected
        assert (
            "! WARNING the reference structure is chiral, check output structures carefully!"
            in result.stdout
        )
        # check all molecules have poses made
        assert "[✓] Pose generation successful for 5/5." in result.stdout
        # make sure stereo filtering is not run
        assert "[✓] Stereochemistry filtering complete" not in result.stdout
        # check the failure warning is not printed
        assert (
            "WARNING some ligands failed to have poses generated see failed_ligands"
            not in result.stdout
        )
        # check we can load the result
        prep_dataset = AlchemyDataSet.from_file(
            "mac1-testing/prepared_alchemy_dataset.json"
        )
        # make sure the receptor is writen to file
        assert (
            pathlib.Path(prep_dataset.dataset_name)
            .joinpath(f"{prep_dataset.reference_complex.target.target_name}.pdb")
            .exists()
        )
        assert prep_dataset.dataset_name == "mac1-testing"
        assert len(prep_dataset.input_ligands) == 5
        assert len(prep_dataset.posed_ligands) == 5
        assert prep_dataset.failed_ligands is None


def test_alchemy_status_all(monkeypatch, alchemiscale_helper):
    """Mock testing the status all command."""

    from alchemiscale import AlchemiscaleClient
    from alchemiscale.models import ScopedKey

    def _get_resource(*args, **kwargs):
        return {"complete": 1, "running": 2, "waiting": 3}

    def get_network_keys(*args, **kwargs):
        """Mock a network key for a running network"""
        return [ScopedKey(gufe_key="fakenetwork", org="asap", campaign="alchemy", project="testing")]

    monkeypatch.setattr(AlchemiscaleClient, "_get_resource", _get_resource)
    monkeypatch.setattr(AlchemiscaleClient, "query_networks", get_network_keys)

    runner = CliRunner()

    result = runner.invoke(
        alchemy,
        [
            "status",
            "-a"
        ]
    )
    assert result.exit_code == 0
    assert "complete                                     │                             1 " in result.stdout
    assert "│ fakenetwork-asap-alchemy-testing │ 1    │ 2    │ 3     │ 0    │ 0     │ 0    │" in result.stdout
