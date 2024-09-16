import pytest
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.ml.cli_mlops import mlops as cli
from click.testing import CliRunner


def test_mlops_run(tmp_path):

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "train-gat-for-endpoint",
            "-p",
            "MERS-CoV-MPro_fluorescence-dose-response_weizmann",
            "-n",
            1,
            "-e",
            1,
            "-o",
            tmp_path,
        ],
    )
    assert result.exit_code == 0
