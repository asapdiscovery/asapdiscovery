"""Tests for the fragalysis data fetching/wrangling/processing"""

import copy
import os
import shutil
import traceback

import pytest
from asapdiscovery.data.cli.cli import data as cli
from asapdiscovery.data.schema.legacy import CrystalCompoundData
from asapdiscovery.data.services.fragalysis.fragalysis_download import (
    API_CALL_BASE_LEGACY,
    BASE_URL_LEGACY,
    FragalysisTargets,
    download,
    parse_fragalysis,
)
from asapdiscovery.data.testing.test_resources import fetch_test_file
from click.testing import CliRunner


def click_success(result):
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0


def fragalysis_api_call(target):
    """Fragalysis API call for downloading target data"""

    api_call = copy.deepcopy(API_CALL_BASE_LEGACY)
    api_call["target_name"] = target
    return api_call


@pytest.mark.skip(reason="Fragalysis call is giving HTTP 500 error.")
class TestFragalysisDownload:
    """Class to test the download of data from Fragalysis."""

    @pytest.mark.parametrize("target", FragalysisTargets.get_values())
    @pytest.mark.parametrize("extract", [True, False])
    def test_download_fragalysis_mpro_zip(self, tmp_path, target, extract):
        """Checks downloading target zip file dataset from fragalysis"""
        api_call = fragalysis_api_call(target)
        zip_file = tmp_path / "fragalysis.zip"
        download(
            zip_file, api_call, extract=extract, base_url=BASE_URL_LEGACY
        )  # don't extract
        assert os.path.exists(zip_file)

    def test_failed_download_fragalysis_target(
        self,
        tmp_path,
    ):
        """Test failed download of target data from fragalysis"""
        from requests import HTTPError

        api_call = fragalysis_api_call("target_name")
        with pytest.raises(HTTPError):
            zip_file = tmp_path / "fragalysis.zip"
            download(zip_file, api_call, extract=False, base_url=BASE_URL_LEGACY)


@pytest.mark.skip(reason="Fragalysis call is giving HTTP 500 error.")
def test_fragalysis_cli(tmpdir):
    with tmpdir.as_cwd():
        runner = CliRunner()
        args = ["download-fragalysis", "-t", "Mpro", "-o", "output.zip", "-x"]
        result = runner.invoke(cli, args)
        assert click_success(result)


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
    xtals = parse_fragalysis(metadata_csv, local_fragalysis)
    assert len(xtals) == 1
    assert type(xtals[0]) is CrystalCompoundData
