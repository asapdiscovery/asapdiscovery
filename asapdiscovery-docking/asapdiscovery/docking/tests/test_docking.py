import os
from pathlib import Path

import pytest
from asapdiscovery.docking.docking_data_validation import DockingResultCols
from asapdiscovery.docking.openeye import POSIT_METHOD, POSITDocker


@pytest.mark.parametrize("posit_method", ["ALL", "HYBRID", "FRED", "SHAPEFIT"])
def test_posit_methods(posit_method):
    assert posit_method in POSIT_METHOD.get_names()
    myvalue = POSIT_METHOD[posit_method]
    from asapdiscovery.docking.openeye import oedocking

    opts = oedocking.OEPositOptions()
    assert opts.SetPositMethods(myvalue.value)


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
class TestDocking:
    def test_docking(self, docking_input_pair):
        docker = POSITDocker(use_omega=False)  # save compute
        results = docker.dock([docking_input_pair])
        assert len(results) == 1
        assert results[0].probability > 0.0

    def test_docking_omega_dense_fails_no_omega(self):
        with pytest.raises(
            ValueError, match="Cannot use omega_dense without use_omega"
        ):
            _ = POSITDocker(use_omega=False, omega_dense=True)

    @pytest.mark.parametrize("use_dask", [True, False])
    @pytest.mark.parametrize("return_for_disk_backend", [True, False])
    def test_docking_dask(
        self,
        docking_input_pair,
        docking_input_pair_simple,
        use_dask,
        return_for_disk_backend,
        tmp_path,
    ):
        docker = POSITDocker(use_omega=False)  # save compute
        results = docker.dock(
            [docking_input_pair, docking_input_pair_simple],
            use_dask=use_dask,
            return_for_disk_backend=return_for_disk_backend,
            output_dir=tmp_path / "docking_results",
        )
        assert len(results) == 2

    def test_docking_with_file_write(self, results_simple, tmp_path):
        docker = POSITDocker(use_omega=False)
        docker.write_docking_files(results_simple, tmp_path)

    def test_docking_with_cache(self, docking_input_pair, tmp_path, caplog):
        import logging

        caplog.set_level(logging.DEBUG)
        docker = POSITDocker(use_omega=False)
        results = docker.dock(
            [docking_input_pair], output_dir=tmp_path / "docking_results"
        )
        assert len(results) == 1
        assert results[0].probability > 0.0
        assert Path(tmp_path / "docking_results").exists()

        results2 = docker.dock(
            [docking_input_pair], output_dir=tmp_path / "docking_results"
        )
        assert len(results2) == 1
        assert results2 == results
        assert "already exists, reading from disk" in caplog.text

    def test_multireceptor_docking(self, docking_multi_structure):
        assert len(docking_multi_structure.complexes) == 2
        docker = POSITDocker(use_omega=False)  # save compute
        results = docker.dock([docking_multi_structure])
        assert len(results) == 1
        assert results[0].input_pair.complex.target.target_name == "Mpro-x1002"
        assert results[0].probability > 0.0

    def test_multipose_docking_with_cache_and_writing(
        self, docking_input_pair, tmp_path, caplog
    ):
        import logging

        caplog.set_level(logging.DEBUG)
        docker = POSITDocker(use_omega=False, num_poses=10)
        results = docker.dock(
            [docking_input_pair], output_dir=tmp_path / "docking_results"
        )

        # although we requested 10 poses, we only get 8
        num_poses_expected = 6
        assert len(results) == num_poses_expected
        assert results[0].probability > 0.0

        results2 = docker.dock(
            [docking_input_pair], output_dir=tmp_path / "docking_results"
        )
        assert len(results2) == num_poses_expected
        results = sorted(results, key=lambda x: x.pose_id)
        results2 = sorted(results2, key=lambda x: x.pose_id)
        assert results2 == results
        assert "already exists, reading from disk" in caplog.text

        for result in results:
            result.write_docking_files(tmp_path / "docking_results")

        assert len(list(tmp_path.glob("docking_results/*/*.pdb"))) == num_poses_expected
        assert (
            len(list(tmp_path.glob("docking_results/*/*.json"))) == num_poses_expected
        )
        assert len(list(tmp_path.glob("docking_results/*/*.sdf"))) == num_poses_expected

    def test_results_to_df(self, results_simple):
        df = results_simple[0].to_df()
        assert DockingResultCols.SMILES in df.columns
        assert DockingResultCols.LIGAND_ID in df.columns
