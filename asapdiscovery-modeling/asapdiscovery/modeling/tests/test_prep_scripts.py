import pytest
from pathlib import Path
from asapdiscovery.data.testing.test_resources import fetch_test_file

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
def mers_seqres():
    return fetch_test_file("mpro_mers_seqres.yaml")


@pytest.fixture
def loop_db():
    return fetch_test_file("fragalysis-mpro_spruce.loop_db")


@pytest.fixture
def ref():
    return fetch_test_file("reference.pdb")


# @pytest.mark.timeout(400)
@pytest.mark.script_launch_mode("subprocess")
def test_mers_download_and_prep(
    script_runner, output_dir, mers_structures, ref, loop_db, mers_seqres
):
    ret = script_runner.run(
        "download-pdbs", "-d", f"{output_dir}", "-p", f"{mers_structures}", "-t", "cif1"
    )
    assert ret.success

    ret = script_runner.run(
        "create-prep-inputs", "-d", f"{output_dir}", "-o", f"{output_dir / 'metadata'}"
    )
    assert ret.success

    ret = script_runner.run(
        "prep-targets",
        "-i",
        f"{output_dir / 'metadata' / 'to_prep.pkl'}",
        "-o",
        f"{output_dir}",
        "-r",
        f"{ref}",
        "-l",
        f"{loop_db}",
        "-s",
        f"{mers_seqres}",
        "-n",
        "4",
    )
    assert ret.success
