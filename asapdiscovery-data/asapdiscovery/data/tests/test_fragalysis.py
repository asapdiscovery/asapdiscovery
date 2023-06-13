"""Tests for the fragalysis data fetching/wrangling/processing"""
import copy
import glob
import os
import shutil

import pytest
from asapdiscovery.data import fragalysis
from asapdiscovery.data.schema import CrystalCompoundData
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.fixture
def mpro_fragalysis_api_call(scope="session"):
    """Fragalysis API call for downloading target data"""
    from asapdiscovery.data.fragalysis import API_CALL_BASE

    api_call = copy.deepcopy(API_CALL_BASE)
    api_call["target_name"] = "Mpro"
    return api_call


@pytest.fixture
def mac1_fragalysis_api_call(scope="session"):
    """Fragalysis API call for downloading target data"""
    from asapdiscovery.data.fragalysis import API_CALL_BASE

    api_call = copy.deepcopy(API_CALL_BASE)
    api_call["target_name"] = "Mac1"
    return api_call


class TestFragalysisDownload:
    """Class to test the download of data from Fragalysis."""

    def test_download_fragalysis_mpro_zip(self, tmp_path, mpro_fragalysis_api_call):
        """Checks downloading target zip file dataset from fragalysis"""
        zip_file = tmp_path / "mpro_fragalysis.zip"
        fragalysis.download(
            zip_file, mpro_fragalysis_api_call, extract=False
        )  # don't extract
        assert os.path.exists(zip_file)

    def test_download_fragalysis_mac1_zip(self, tmp_path, mac1_fragalysis_api_call):
        """Checks downloading target zip file dataset from fragalysis"""
        zip_file = tmp_path / "mac1_fragalysis.zip"
        fragalysis.download(
            zip_file, mac1_fragalysis_api_call, extract=False
        )  # don't extract
        assert os.path.exists(zip_file)

    def test_failed_download_fragalysis_target(
        self, tmp_path, mpro_fragalysis_api_call
    ):
        """Test failed download of target data from fragalysis"""
        from requests import HTTPError

        mpro_fragalysis_api_call["target_name"] = "ThisIsNotATargetName"
        with pytest.raises(HTTPError):
            zip_file = tmp_path / "fragalysis.zip"
            fragalysis.download(zip_file, mpro_fragalysis_api_call)

    def test_sdfs_pdbs_fragalysis_download(self, tmp_path, mpro_fragalysis_api_call):
        """Test SDF and PDB files exists in downloaded fragalysis zip file"""
        zip_file = tmp_path / "mpro_fragalysis.zip"
        fragalysis.download(zip_file, mpro_fragalysis_api_call, extract=True)  # extract
        # Make sure there are sdf and pdb files in the extracted files
        assert glob.glob(
            f"{zip_file.parent}/**/*.sdf", recursive=True
        ), "No SDF files found on extracted fragalysis target zip."
        assert glob.glob(
            f"{zip_file.parent}/**/*.pdb", recursive=True
        ), "No PDB files found on extracted fragalysis target zip."


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


def test_parse_fragalysis(metadata_csv, local_fragalysis):
    xtals = fragalysis.parse_fragalysis(metadata_csv, local_fragalysis)
    assert len(xtals) == 1
    assert type(xtals[0]) == CrystalCompoundData


def test_parse_fragalysis_script(
    script_runner, tmp_path, metadata_csv, local_fragalysis
):
    ret = script_runner.run(
        [
            "fragalysis-to-schema",
            "--metadata_csv",
            f"{metadata_csv}",
            "--aligned_dir",
            f"{local_fragalysis}",
            "-o",
            f"{tmp_path}",
        ]
    )
    out_path = tmp_path / "fragalysis.csv"
    assert ret.success
    assert out_path.exists()
