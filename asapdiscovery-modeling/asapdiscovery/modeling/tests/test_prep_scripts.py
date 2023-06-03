from pathlib import Path

import pytest
from asapdiscovery.data.testing.test_resources import fetch_test_file
import shutil


# This needs to have a scope of session so that a new tmp file is not created for each test
@pytest.fixture(scope="session")
def output_dir(tmp_path_factory, local_path):
    if not type(local_path) == str:
        return tmp_path_factory.mktemp("test_prep")
    else:
        local_path = Path(local_path)
        local_path.mkdir(exist_ok=True)
        assert local_path.exists()
        return local_path


@pytest.fixture
def mers_structures():
    return fetch_test_file("mers-structures.yaml")


@pytest.fixture
def seqres_dict():
    return {
        "mers": fetch_test_file("mpro_mers_seqres.yaml"),
        "sars": fetch_test_file("mpro_sars2_seqres.yaml"),
    }


@pytest.fixture
def loop_db():
    return fetch_test_file("fragalysis-mpro_spruce.loop_db")


@pytest.fixture
def ref():
    return fetch_test_file("reference.pdb")


def test_mers_download_and_create_prep_inputs(
    script_runner, output_dir, mers_structures
):
    ret = script_runner.run(
        "download-pdbs",
        "-d",
        f"{output_dir / 'input_structures'}",
        "-p",
        f"{mers_structures}",
        "-t",
        "cif1",
    )
    assert ret.success

    ret = script_runner.run(
        "create-prep-inputs",
        "-d",
        f"{output_dir / 'input_structures'}",
        "-o",
        f"{output_dir / 'metadata'}",
        "--components_to_keep",
        "protein",
        "--active_site",
        "HIS:41: :A:0: ",
    )
    assert ret.success


@pytest.mark.timeout(400)
@pytest.mark.script_launch_mode("subprocess")
@pytest.mark.skip(
    reason="This test takes too long to run and timeout doesn't seem to work"
)
def test_mers_prep(script_runner, output_dir, ref, loop_db, seqres_dict):
    ret = script_runner.run(
        "prep-targets",
        "-i",
        f"{output_dir / 'metadata' / 'to_prep.pkl'}",
        "-o",
        f"{output_dir / 'prepped_structures'}",
        "-r",
        f"{ref}",
        "-l",
        f"{loop_db}",
        "-s",
        f"{seqres_dict['mers']}",
        "-n",
        "4",
    )
    assert ret.success


# TODO: This code block is copied from test_fragalysis
#  I think we should be able to use the same fixtures for both tests
@pytest.fixture
def metadata_csv():
    return fetch_test_file("metadata.csv")


@pytest.fixture
def local_fragalysis(tmp_path):
    pdb = fetch_test_file("Mpro-P2660_0A_bound.pdb")
    new_path = tmp_path / "aligned/Mpro-P2660_0A"
    new_path.mkdir(parents=True)
    shutil.copy(pdb, new_path / "Mpro-P2660_0A_bound.pdb")
    return new_path.parent


# TODO: End copied code block


def test_sars_create_prep_inputs(
    script_runner, output_dir, metadata_csv, local_fragalysis
):
    ret = script_runner.run(
        [
            "fragalysis-to-schema",
            "--metadata_csv",
            f"{metadata_csv}",
            "--aligned_dir",
            f"{local_fragalysis}",
            "-o",
            f"{output_dir / 'metadata'}",
        ]
    )
    out_path = output_dir / "metadata/fragalysis.csv"
    assert ret.success
    assert out_path.exists()

    ret = script_runner.run(
        [
            "create-prep-inputs",
            "-i",
            f"{out_path}",
            "-o",
            f"{output_dir / 'metadata'}",
            "--components_to_keep",
            "protein",
            "ligand",
        ]
    )
    assert ret.success
    assert (output_dir / "metadata" / "to_prep.pkl").exists()


@pytest.mark.timeout(400)
@pytest.mark.script_launch_mode("subprocess")
@pytest.mark.skip(
    reason="This test takes too long to run and timeout doesn't seem to work"
)
def test_sars_prep(script_runner, output_dir, ref, loop_db, seqres_dict):
    ret = script_runner.run(
        "prep-targets",
        "-i",
        f"{output_dir / 'metadata' / 'to_prep.pkl'}",
        "-o",
        f"{output_dir / 'prepped_structures'}",
        "-r",
        f"{ref}",
        "-l",
        f"{loop_db}",
        "-s",
        f"{seqres_dict['sars']}",
        "-n",
        "4",
    )
    assert ret.success
