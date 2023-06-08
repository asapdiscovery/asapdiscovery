import shutil

import pytest
from asapdiscovery.data.testing.test_resources import fetch_test_file


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
        "-s",
        f"{output_dir / 'input_structures/rcsb_8czt-assembly1.cif'}",
        "-o",
        f"{output_dir / 'metadata'}",
        "--components_to_keep",
        "protein",
        "--oe_active_site_residue",
        "HIS:41: :A: ",
    )
    assert ret.success


@pytest.mark.timeout(300)
@pytest.mark.script_launch_mode("subprocess")
def test_mers_prep(script_runner, output_dir, ref, loop_db, seqres_dict):
    ret = script_runner.run(
        "prep-targets",
        "-i",
        f"{output_dir / 'metadata' / 'to_prep.json'}",
        "-o",
        f"{output_dir / 'prepped_structures'}",
        "-r",
        f"{ref}",
        "-l",
        f"{loop_db}",
        "-s",
        f"{seqres_dict['mers']}",
        "-n",
        "1",
        "--debug_num",
        "1",
    )
    assert ret.success


# TODO: This code block is mostly copied from test_fragalysis
#  I think we should be able to use the same fixtures for both tests but idk how
@pytest.fixture
def metadata_csv():
    return fetch_test_file("metadata.csv")


@pytest.fixture
def structure_file(tmp_path):
    return fetch_test_file("Mpro-P2660_0A_bound.pdb")


@pytest.fixture
def local_fragalysis(tmp_path, structure_file):
    new_path = tmp_path / "aligned/Mpro-P2660_0A"
    new_path.mkdir(parents=True)
    shutil.copy(structure_file, new_path / "Mpro-P2660_0A_bound.pdb")
    return new_path.parent


# TODO: End copied code block


@pytest.mark.parametrize("serialized_input", [True, False])
def test_sars_create_prep_inputs(
    script_runner,
    output_dir,
    metadata_csv,
    local_fragalysis,
    structure_file,
    serialized_input,
):
    output_dir = output_dir if serialized_input else output_dir / "non_serialized_input"
    if serialized_input:
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
        assert (output_dir / "metadata" / "to_prep.json").exists()
    else:
        ret = script_runner.run(
            [
                "create-prep-inputs",
                "--structure_file",
                f"{structure_file}",
                "-o",
                f"{output_dir / 'metadata'}",
                "--components_to_keep",
                "protein",
                "ligand",
            ]
        )
        assert ret.success
        assert (output_dir / "metadata" / "to_prep.json").exists()


@pytest.mark.timeout(300)
@pytest.mark.script_launch_mode("subprocess")
def test_sars_prep(script_runner, output_dir, ref, loop_db, seqres_dict):
    ret = script_runner.run(
        "prep-targets",
        "-i",
        f"{output_dir / 'metadata' / 'to_prep.json'}",
        "-o",
        f"{output_dir / 'prepped_structures'}",
        "-r",
        f"{ref}",
        "-l",
        f"{loop_db}",
        "-s",
        f"{seqres_dict['sars']}",
        "-n",
        "1",
        "--debug_num",
        "1",
    )
    assert ret.success
