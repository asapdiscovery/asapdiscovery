import pathlib

import pandas as pd
import pytest
from alchemiscale import AlchemiscaleClient
from alchemiscale.models import ScopedKey
from asapdiscovery.alchemy.cli.cli import alchemy
from asapdiscovery.alchemy.schema.fec import (
    FreeEnergyCalculationFactory,
    FreeEnergyCalculationNetwork,
)
from asapdiscovery.alchemy.schema.prep_workflow import (
    AlchemyDataSet,
    AlchemyPrepWorkflow,
)
from asapdiscovery.data.services.cdd.cdd_api import CDDAPI
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


def test_alchemy_prep_run_with_fails_and_charges(
    tmpdir, mac1_complex, openeye_charged_prep_workflow
):
    """Test running the alchemy prep workflow on a set of mac1 ligands and that failures are captured"""

    # locate the ligands input file
    ligand_file = fetch_test_file("constrained_conformer/mac1_ligands.smi")

    runner = CliRunner()

    with tmpdir.as_cwd():
        # complex to a local file
        mac1_complex.to_json_file("complex.json")
        # write out the workflow to file
        openeye_charged_prep_workflow.to_file("openeye_workflow.json")

        result = runner.invoke(
            alchemy,
            [
                "prep",
                "run",
                "-f",
                "openeye_workflow.json",
                "-n",
                "mac1-testing",
                "-l",
                ligand_file.as_posix(),
                "-r",
                "complex.json",
                "-p",
                1,
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
        assert "WARNING 2 ligands failed to have poses generated" in result.stdout
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
        # make sure the csv of failed ligands is writen to file
        assert (
            pathlib.Path(prep_dataset.dataset_name)
            .joinpath("failed_ligands.csv")
            .exists()
        )
        # check the dataset details are as expected
        assert prep_dataset.dataset_name == "mac1-testing"
        assert len(prep_dataset.input_ligands) == 5
        assert len(prep_dataset.posed_ligands) == 3
        assert len(prep_dataset.failed_ligands["InconsistentStereo"]) == 2
        for ligand in prep_dataset.posed_ligands:
            assert "atom.dprop.PartialCharge" in ligand.tags
            assert ligand.charge_provenance is not None


def test_alchemy_prep_run_all_pass(tmpdir, mac1_complex, openeye_prep_workflow):
    """Test running the alchemy prep workflow and make sure all ligands pass when expected."""

    # locate the ligands input file
    ligand_file = fetch_test_file("constrained_conformer/mac1_ligands.smi")

    runner = CliRunner()

    with tmpdir.as_cwd():
        # complex to a local file
        mac1_complex.to_json_file("complex.json")
        # create a new prep workflow which allows incorrect stereo
        workflow = openeye_prep_workflow.copy(deep=True)
        workflow.strict_stereo = False
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
                "-p",
                1,
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
            "WARNING 2 ligands failed to have poses generated see failed_ligands"
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


def test_alchemy_prep_receptor_pick(tmpdir, mac1_complex, openeye_prep_workflow):
    """Test running the alchemy prep workflow and letting it select the receptor."""

    # locate the ligands input file
    ligand_file = fetch_test_file("constrained_conformer/mac1_ligands.smi")
    runner = CliRunner()

    with tmpdir.as_cwd():
        # store the complex in the cache folder we only have one so it should select this one
        receptor_cache = pathlib.Path("cache")
        receptor_cache.mkdir(exist_ok=True)

        mac1_complex.to_json_file(receptor_cache.joinpath("complex.json"))
        # create a new prep workflow with no expansion and only valid stereo
        workflow = openeye_prep_workflow.copy(deep=True)
        workflow.strict_stereo = True
        workflow.stereo_expander = None
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
                "-sd",
                "cache",
                "-p",
                1,
            ],
        )
        assert result.exit_code == 0
        # make sure that we are selecting a receptor
        assert (
            "Selected SARS2_Mac1A_A1496-ASAP-0008674-001 as the best reference "
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
        assert "[✓] Stereochemistry filtering complete" in result.stdout
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


def test_alchemy_prep_run_from_postera(
    tmpdir, mac1_complex, openeye_prep_workflow, monkeypatch
):
    """Test running the alchemy prep workflow on a set of mac1 ligands downloaded from postera."""
    from asapdiscovery.alchemy.cli import utils
    from asapdiscovery.data.readers.molfile import MolFileFactory
    from asapdiscovery.data.schema.ligand import Ligand

    # locate the ligands input file
    ligand_file = fetch_test_file("constrained_conformer/mac1_ligands.smi")

    # Mock the method to download from postera making sure the molecule set name is passed
    def pull(molecule_set_name: str) -> list[Ligand]:
        assert molecule_set_name == "mac1_ligands"
        return MolFileFactory(filename=ligand_file.as_posix()).load()

    monkeypatch.setattr(utils, "pull_from_postera", pull)

    runner = CliRunner()

    with tmpdir.as_cwd():
        # complex to a local file
        mac1_complex.to_json_file("complex.json")
        # write out the workflow to file
        openeye_prep_workflow.to_file("openeye_workflow.json")

        result = runner.invoke(
            alchemy,
            [
                "prep",
                "run",
                "-f",
                "openeye_workflow.json",
                "-n",
                "mac1-testing",
                "-pm",
                "mac1_ligands",
                "-r",
                "complex.json",
                "-p",
                1,
            ],
        )
        assert result.exit_code == 0


def test_alchemy_status_all(monkeypatch):
    """Mock testing the status all command."""
    monkeypatch.setenv("ALCHEMISCALE_ID", "my-id")
    monkeypatch.setenv("ALCHEMISCALE_KEY", "my-key")

    network_key = ScopedKey(
        gufe_key="fakenetwork",
        org="asap",
        campaign="alchemy",
        project="testing",
    )
    network_status = {"complete": 1, "running": 2, "waiting": 3}

    def get_networks_status(*args, **kwargs):
        """ ""We mock this as it changes the return on get scope status and get network status"""
        return [network_status]

    def _get_resource(*args, **kwargs):
        return network_status

    def get_network_keys(*args, **kwargs):
        """Mock a network key for a running network"""
        return [
            network_key,
        ]

    def get_actioned(*args, **kwargs):
        assert network_key in kwargs["networks"]
        return [
            [i for i in range(5)],
        ]

    def get_networks_weight(*args, **kwargs):
        """Mock the network priority weight"""
        return [
            1,
        ]

    # mock the full call stack in the helper function
    monkeypatch.setattr(AlchemiscaleClient, "get_networks_status", get_networks_status)
    monkeypatch.setattr(AlchemiscaleClient, "query_networks", get_network_keys)
    monkeypatch.setattr(AlchemiscaleClient, "get_networks_actioned_tasks", get_actioned)
    monkeypatch.setattr(AlchemiscaleClient, "get_networks_weight", get_networks_weight)
    monkeypatch.setattr(AlchemiscaleClient, "_get_resource", _get_resource)

    runner = CliRunner()

    result = runner.invoke(alchemy, ["status", "-a"])
    assert result.exit_code == 0
    assert (
        "complete                                     │                             1 "
        in result.stdout
    )
    assert (
        "│ fakenetwork-asap-alchemy-testing │ 1   │ 2  │ 3   │ 0  │ 0   │ 0  │ 5   │ 1  │"
        in result.stdout
    )


def test_alchemy_stop(monkeypatch):
    """Test canceling the actioned tasks on a network"""
    monkeypatch.setenv("ALCHEMISCALE_ID", "my-id")
    monkeypatch.setenv("ALCHEMISCALE_KEY", "my-key")

    runner = CliRunner()

    network_key = ScopedKey(
        gufe_key="fakenetwork-12345",
        org="asap",
        campaign="alchemy",
        project="testing",
    )

    def get_network_actioned_tasks(*args, **kwargs):
        assert ScopedKey.from_str(kwargs["network"]) == network_key
        return [1, 2, 3, 4]

    def cancel_tasks(*args, **kwargs):
        tasks = kwargs["tasks"]
        network = ScopedKey.from_str(kwargs["network"])
        assert network == network_key
        return tasks

    monkeypatch.setattr(
        AlchemiscaleClient, "get_network_actioned_tasks", get_network_actioned_tasks
    )
    monkeypatch.setattr(AlchemiscaleClient, "cancel_tasks", cancel_tasks)

    result = runner.invoke(alchemy, ["stop", "-nk", network_key])
    assert result.exit_code == 0
    assert (
        "Canceled 4 actioned tasks for network fakenetwork-12345-asap-alchemy-testing"
        in result.stdout
    )


def test_submit_bad_campaign(tyk2_fec_network, tmpdir):
    """Make sure an error is raised if the org is asap but the campaign is not in public or confidential."""

    runner = CliRunner()

    with tmpdir.as_cwd():
        # write the network to a local file
        tyk2_fec_network.to_file("planned_network.json")
        with pytest.raises(ValueError) as exp:
            _ = runner.invoke(
                alchemy,
                ["submit", "-o", "asap", "-c", "fancy_campaign", "-p", "fancy_ligands"],
                catch_exceptions=False,
            )
            # cannot use match here due to regex escaping
            assert (
                exp.value.args[0]
                == "If organization (`-o`) is set to 'asap' (default), campaign (`-c`) must be either of 'public' or 'confidential'."
            )


def test_alchemy_predict_no_experimental_data(tyk2_result_network, tmpdir):
    """Test predicting the absolute and relative free energies with no experimental data, interactive reports should
    not be generated in this mode.
    We also test that a warning is printed in the terminal as the target is missing so the results can not be uploaded
    to postera.
    """

    runner = CliRunner()

    with tmpdir.as_cwd():
        # write the results file to local
        tyk2_result_network.to_file("result_network.json")

        result = runner.invoke(
            alchemy,
            ["predict", "-pm", "my-molset"],
        )
        assert result.exit_code == 0
        assert "Loaded FreeEnergyCalculationNetwork from" in result.stdout
        assert "Absolute predictions written" in result.stdout
        assert "Relative predictions written" in result.stdout
        assert (
            "WARNING a postera molecule set name was provided without a target, results "
            in result.stdout
        )
        # load the datasets and check the results match what's expected
        absolute_dataframe = pd.read_csv("predictions-absolute-tyk2-small-test.csv")

        mol_data = absolute_dataframe.iloc[0]
        assert mol_data["SMILES"] == "CC(=O)Nc1cc(ccn1)NC(=O)c2c(cccc2Cl)Cl"
        assert mol_data["Inchi_Key"] == "DKNAYSZNMZIMIZ-UHFFFAOYSA-N"
        assert mol_data["label"] == "lig_ejm_31"
        assert mol_data["DG (kcal/mol) (FECS)"] == pytest.approx(-0.1332, abs=1e-4)
        assert mol_data["uncertainty (kcal/mol) (FECS)"] == pytest.approx(
            0.0757, abs=1e-4
        )

        relative_dataframe = pd.read_csv("predictions-relative-tyk2-small-test.csv")
        relative_mol_data = relative_dataframe.iloc[0]
        assert relative_mol_data["SMILES_A"] == "CC(=O)Nc1cc(ccn1)NC(=O)c2c(cccc2Cl)Cl"
        assert (
            relative_mol_data["SMILES_B"]
            == "c1cc(c(c(c1)Cl)C(=O)Nc2ccnc(c2)NC(=O)C3CCC3)Cl"
        )
        assert relative_mol_data["Inchi_Key_A"] == "DKNAYSZNMZIMIZ-UHFFFAOYSA-N"
        assert relative_mol_data["Inchi_Key_B"] == "YJMGZFGQBBEAQT-UHFFFAOYSA-N"
        assert relative_mol_data["labelA"] == "lig_ejm_31"
        assert relative_mol_data["labelB"] == "lig_ejm_47"
        assert relative_mol_data["DDG (kcal/mol) (FECS)"] == pytest.approx(
            0.1115, abs=1e-4
        )
        assert relative_mol_data["uncertainty (kcal/mol) (FECS)"] == pytest.approx(
            0.1497, abs=1e-4
        )


def test_alchemy_predict_experimental_data(
    tyk2_result_network, tmpdir, tyk2_reference_data
):
    """
    Test predicting the absolute and relative free energies with experimental data, the predictions should be shifted
    by the experimental mean to get them in the correct energy range and the interactive reports should be generated.
    """

    runner = CliRunner()

    with tmpdir.as_cwd():
        # write the results file to local
        tyk2_result_network.to_file("result_network.json")

        result = runner.invoke(
            alchemy, ["predict", "-rd", tyk2_reference_data, "-ru", "IC50"]
        )
        assert result.exit_code == 0
        assert "Loaded FreeEnergyCalculationNetwork" in result.stdout
        assert (
            "Absolute report written to predictions-absolute-tyk2-small-test.html"
            in result.stdout
        )
        assert (
            "Relative report written to predictions-relative-tyk2-small-test.html"
            in result.stdout
        )
        # load the datasets and check the results match what's expected
        absolute_dataframe = pd.read_csv("predictions-absolute-tyk2-small-test.csv")
        mol_data = absolute_dataframe.iloc[0]
        assert mol_data["SMILES"] == "CC(=O)Nc1cc(ccn1)NC(=O)c2c(cccc2Cl)Cl"
        assert mol_data["Inchi_Key"] == "DKNAYSZNMZIMIZ-UHFFFAOYSA-N"
        assert mol_data["label"] == "lig_ejm_31"
        # make sure the results have been shifted to match the experimental range
        assert mol_data["DG (kcal/mol) (FECS)"] == pytest.approx(-10.2182, abs=1e-4)
        assert mol_data["uncertainty (kcal/mol) (FECS)"] == pytest.approx(
            0.0757, abs=1e-4
        )
        # make sure the experimental data has been added
        assert mol_data["DG (kcal/mol) (EXPT)"] == pytest.approx(-9.5739, abs=1e-4)
        # make sure the prediction error has been calculated
        assert mol_data["prediction error (kcal/mol)"] == pytest.approx(
            0.6443, abs=1e-4
        )

        relative_dataframe = pd.read_csv("predictions-relative-tyk2-small-test.csv")
        relative_mol_data = relative_dataframe.iloc[0]
        assert relative_mol_data["SMILES_A"] == "CC(=O)Nc1cc(ccn1)NC(=O)c2c(cccc2Cl)Cl"
        assert (
            relative_mol_data["SMILES_B"]
            == "c1cc(c(c(c1)Cl)C(=O)Nc2ccnc(c2)NC(=O)C3CCC3)Cl"
        )
        assert relative_mol_data["Inchi_Key_A"] == "DKNAYSZNMZIMIZ-UHFFFAOYSA-N"
        assert relative_mol_data["Inchi_Key_B"] == "YJMGZFGQBBEAQT-UHFFFAOYSA-N"
        assert relative_mol_data["labelA"] == "lig_ejm_31"
        assert relative_mol_data["labelB"] == "lig_ejm_47"
        # these should not be changed as they do not need shifting
        assert relative_mol_data["DDG (kcal/mol) (FECS)"] == pytest.approx(
            0.1115, abs=1e-4
        )
        assert relative_mol_data["uncertainty (kcal/mol) (FECS)"] == pytest.approx(
            0.1497, abs=1e-4
        )
        # make sure the experimental data has been added
        assert relative_mol_data["DDG (kcal/mol) (EXPT)"] == pytest.approx(
            -0.1542, abs=1e-4
        )
        assert relative_mol_data["prediction error (kcal/mol)"] == pytest.approx(
            0.2657, abs=1e-4
        )


def test_alchemy_predict_ccd_data(
    tmpdir, tyk2_result_network, tyk2_reference_data, monkeypatch
):
    """
    Make sure we can do a prediction with experimental data when using the CDD api interface.

    Notes:
        The CDD api will be mocked, so we are testing the ability to match the data up correctly.
        We expected slightly different values here due to rounding of the pIC50 via this pathway.
    """

    # mock the env variables
    monkeypatch.setenv("CDD_API_KEY", "mykey")
    monkeypatch.setenv("CDD_VAULT_NUMBER", "1")
    protocol_name = "my-protocol"

    # mock the cdd_api
    def get_tyk2_data(*args, **kwargs):
        data = pd.read_csv(tyk2_reference_data, index_col=0)
        # format the data and add expected columns
        ic50_lower, ic50_upper, curve, inchi, inchi_key = [], [], [], [], []
        data.rename(
            columns={
                "SMILES": "Smiles",
                "IC50_GMean (µM)": f"{protocol_name}: IC50 (µM)",
            },
            inplace=True,
        )
        data.drop(columns=["IC50_GMean (µM) Standard Deviation (×/÷)"], inplace=True)
        for _, row in data.iterrows():
            # calculate the required data
            rdkit_mol = Chem.MolFromSmiles(row["Smiles"])
            inchi.append(Chem.MolToInchi(rdkit_mol))
            inchi_key.append(Chem.MolToInchiKey(rdkit_mol))
            curve.append(1.1)
            ic50_lower.append(row[f"{protocol_name}: IC50 (µM)"] - 0.001)
            ic50_upper.append(row[f"{protocol_name}: IC50 (µM)"] + 0.01)
        data[f"{protocol_name}: IC50 CI (Lower) (µM)"] = ic50_lower
        data[f"{protocol_name}: IC50 CI (Upper) (µM)"] = ic50_upper
        data[f"{protocol_name}: Curve class"] = curve
        data["Inchi"] = inchi
        data["Inchi Key"] = inchi_key
        # these should be the cdd ids but just use some mock id
        data["name"] = data["Molecule Name"]
        return data

    monkeypatch.setattr(CDDAPI, "get_ic50_data", get_tyk2_data)

    runner = CliRunner()

    with tmpdir.as_cwd():
        # write the results file to local
        tyk2_result_network.to_file("result_network.json")

        result = runner.invoke(alchemy, ["predict", "-ep", protocol_name])
        assert result.exit_code == 0
        assert "Loaded FreeEnergyCalculationNetwork" in result.stdout
        assert (
            "Absolute report written to predictions-absolute-tyk2-small-test.html"
            in result.stdout
        )
        assert (
            "Relative report written to predictions-relative-tyk2-small-test.html"
            in result.stdout
        )
        # load the datasets and check the results match what's expected
        absolute_dataframe = pd.read_csv("predictions-absolute-tyk2-small-test.csv")
        # make sure all results are present
        assert len(absolute_dataframe) == 10
        mol_data = absolute_dataframe.iloc[0]
        assert mol_data["SMILES"] == "CC(=O)Nc1cc(ccn1)NC(=O)c2c(cccc2Cl)Cl"
        assert mol_data["Inchi_Key"] == "DKNAYSZNMZIMIZ-UHFFFAOYSA-N"
        assert mol_data["label"] == "lig_ejm_31"
        # make sure the results have been shifted to match the experimental range
        assert mol_data["DG (kcal/mol) (FECS)"] == pytest.approx(-10.2151, abs=1e-4)
        assert mol_data["uncertainty (kcal/mol) (FECS)"] == pytest.approx(
            0.0757, abs=1e-4
        )
        # make sure the experimental data has been added
        assert mol_data["DG (kcal/mol) (EXPT)"] == pytest.approx(-9.5721, abs=1e-4)
        # make sure the prediction error has been calculated
        assert mol_data["prediction error (kcal/mol)"] == pytest.approx(
            0.6429, abs=1e-4
        )

        relative_dataframe = pd.read_csv("predictions-relative-tyk2-small-test.csv")
        # make sure all results are present
        assert len(relative_dataframe) == 9
        relative_mol_data = relative_dataframe.iloc[0]
        assert relative_mol_data["SMILES_A"] == "CC(=O)Nc1cc(ccn1)NC(=O)c2c(cccc2Cl)Cl"
        assert (
            relative_mol_data["SMILES_B"]
            == "c1cc(c(c(c1)Cl)C(=O)Nc2ccnc(c2)NC(=O)C3CCC3)Cl"
        )
        assert relative_mol_data["Inchi_Key_A"] == "DKNAYSZNMZIMIZ-UHFFFAOYSA-N"
        assert relative_mol_data["Inchi_Key_B"] == "YJMGZFGQBBEAQT-UHFFFAOYSA-N"
        assert relative_mol_data["labelA"] == "lig_ejm_31"
        assert relative_mol_data["labelB"] == "lig_ejm_47"
        # these should not be changed as they do not need shifting
        assert relative_mol_data["DDG (kcal/mol) (FECS)"] == pytest.approx(
            0.1115, abs=1e-4
        )
        assert relative_mol_data["uncertainty (kcal/mol) (FECS)"] == pytest.approx(
            0.1497, abs=1e-4
        )
        # make sure the experimental data has been added
        assert relative_mol_data["DDG (kcal/mol) (EXPT)"] == pytest.approx(
            -0.1499, abs=1e-4
        )
        assert relative_mol_data["prediction error (kcal/mol)"] == pytest.approx(
            0.2615, abs=1e-4
        )


def test_predict_missing_all_exp_data(
    tyk2_reference_data, tyk2_result_network, tmpdir, monkeypatch
):
    """
    Test making a prediction when experimental data is provided but does not overlap with the ligands, this should
    stop the generation of the interactive reports.
    """
    # mock the env variables
    monkeypatch.setenv("CDD_API_KEY", "mykey")
    monkeypatch.setenv("CDD_VAULT_NUMBER", "1")
    protocol_name = "my-protocol"

    # mock the cdd_api
    def get_tyk2_data(*args, **kwargs):
        methanol = Chem.MolFromSmiles("CO")
        fake_tyk2_data = {
            "Smiles": "CO",
            f"{protocol_name}: IC50 (µM)": 50,
            f"{protocol_name}: IC50 CI (Lower) (µM)": 49.98,
            f"{protocol_name}: IC50 CI (Upper) (µM)": 50.001,
            f"{protocol_name}: Curve class": 1.1,
            "Inchi": Chem.MolToInchi(methanol),
            "Inchi Key": Chem.MolToInchiKey(methanol),
            "name": "methanol",
            "Molecule Name": "methanol",
        }

        return pd.DataFrame([fake_tyk2_data])

    monkeypatch.setattr(CDDAPI, "get_ic50_data", get_tyk2_data)

    runner = CliRunner()

    with tmpdir.as_cwd():
        # write the results file to local
        tyk2_result_network.to_file("result_network.json")

        result = runner.invoke(alchemy, ["predict", "-ep", protocol_name])
        assert result.exit_code == 0
        assert "Loaded FreeEnergyCalculationNetwork" in result.stdout
        assert (
            "Absolute report written to predictions-absolute-tyk2-small-test.html"
            not in result.stdout
        )
        assert (
            "Relative report written to predictions-relative-tyk2-small-test.html"
            not in result.stdout
        )
        # load the datasets and check the results match what's expected
        absolute_dataframe = pd.read_csv("predictions-absolute-tyk2-small-test.csv")
        # make sure all results are present
        assert len(absolute_dataframe) == 10
        mol_data = absolute_dataframe.iloc[0]
        assert mol_data["SMILES"] == "CC(=O)Nc1cc(ccn1)NC(=O)c2c(cccc2Cl)Cl"
        assert mol_data["Inchi_Key"] == "DKNAYSZNMZIMIZ-UHFFFAOYSA-N"
        assert mol_data["label"] == "lig_ejm_31"
        # make sure the results have not been shifted
        assert mol_data["DG (kcal/mol) (FECS)"] == pytest.approx(-0.1332, abs=1e-4)
        assert mol_data["uncertainty (kcal/mol) (FECS)"] == pytest.approx(
            0.0757, abs=1e-4
        )

        relative_dataframe = pd.read_csv("predictions-relative-tyk2-small-test.csv")
        # make sure all results are present
        assert len(relative_dataframe) == 9
        relative_mol_data = relative_dataframe.iloc[0]
        assert relative_mol_data["SMILES_A"] == "CC(=O)Nc1cc(ccn1)NC(=O)c2c(cccc2Cl)Cl"
        assert (
            relative_mol_data["SMILES_B"]
            == "c1cc(c(c(c1)Cl)C(=O)Nc2ccnc(c2)NC(=O)C3CCC3)Cl"
        )
        assert relative_mol_data["Inchi_Key_A"] == "DKNAYSZNMZIMIZ-UHFFFAOYSA-N"
        assert relative_mol_data["Inchi_Key_B"] == "YJMGZFGQBBEAQT-UHFFFAOYSA-N"
        assert relative_mol_data["labelA"] == "lig_ejm_31"
        assert relative_mol_data["labelB"] == "lig_ejm_47"
        # these should not be changed as they do not need shifting
        assert relative_mol_data["DDG (kcal/mol) (FECS)"] == pytest.approx(
            0.1115, abs=1e-4
        )
        assert relative_mol_data["uncertainty (kcal/mol) (FECS)"] == pytest.approx(
            0.1497, abs=1e-4
        )


def test_predict_wrong_units(tyk2_result_network, tyk2_reference_data, tmpdir):
    """Make sure an error is raised if the units can not be found in the csv headings"""

    runner = CliRunner()

    with tmpdir.as_cwd():
        # write the results file to local
        tyk2_result_network.to_file("result_network.json")
        with pytest.raises(
            RuntimeError,
            match="Could not determine the assay tag from the provided units pIC50",
        ):
            # use the wrong unit heading
            runner.invoke(
                alchemy,
                ["predict", "-rd", tyk2_reference_data, "-ru", "pIC50"],
                catch_exceptions=False,  # let the exception buble up so pytest can check it
            )
