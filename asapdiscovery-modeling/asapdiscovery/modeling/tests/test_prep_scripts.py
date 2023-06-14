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


class TestCreatePrepInputs:

    CREATE_PREP_INPUT_ARGS = [
        "create-prep-inputs",
        "--structure_file",
        f"{structure_file}",
        "--components_to_keep",
        "protein",
        "ligand",
    ]

    @pytest.mark.parametrize(
        "input_type", ["--structure_file", "--structure_dir", "--input_file"]
    )
    def test_create_prep_inputs(
        self,
        script_runner,
        input_type,
        local_fragalysis,
        structure_file,
        output_dir,
        metadata_csv,
    ):
        input_args = self.CREATE_PREP_INPUT_ARGS + ["--output_dir", f"{output_dir}"]
        if input_type == "--structure_file":
            input_args += [input_type, f"{structure_file}"]
        elif input_type == "--structure_dir":
            input_args += [input_type, f"{local_fragalysis}"]
        elif input_type == "--input_file":
            # We need to make the schema first
            script_runner.run(
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
            serialized_fragalysis = output_dir / "metadata/fragalysis.csv"

            input_args += [input_type, f"{serialized_fragalysis}"]

        ret = script_runner.run(*input_args)
        assert ret.success
        assert (output_dir / "to_prep.json").exists()


class TestPrepFromRCSB:
    @pytest.mark.timeout(300)
    @pytest.mark.script_launch_mode("subprocess")
    def test_mers_prep(
        self, script_runner, output_dir, mers_structures, ref, loop_db, seqres_dict
    ):
        # First, download the structures
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

        # Then, create the inputs for prep
        ret = script_runner.run(
            "create-prep-inputs",
            "-s",
            f"{output_dir / 'input_structures/rcsb_8czt-assembly1.cif'}",
            "-o",
            f"{output_dir / 'metadata'}",
            "--components_to_keep",
            "protein",
            "--oe_active_site_residue",
            "HIS:41: :A:1",
        )
        assert ret.success

        # Then, run prep
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


class TestPrepFromFragalysis:
    @pytest.mark.timeout(300)
    @pytest.mark.script_launch_mode("subprocess")
    def test_sars_prep(
        self,
        script_runner,
        output_dir,
        metadata_csv,
        local_fragalysis,
        ref,
        loop_db,
        seqres_dict,
        structure_file,
    ):
        # First, convert fragalysis to schema
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
        serialized_fragalysis = output_dir / "metadata/fragalysis.csv"
        assert ret.success
        assert serialized_fragalysis.exists()

        # Then, create the inputs for prep
        ret = script_runner.run(
            [
                "create-prep-inputs",
                "-i",
                f"{serialized_fragalysis}",
                "-o",
                f"{output_dir / 'metadata'}",
                "--components_to_keep",
                "protein",
                "ligand",
            ]
        )
        serialized_prep_targets = output_dir / "metadata" / "to_prep.json"
        assert ret.success
        assert serialized_prep_targets.exists()

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
