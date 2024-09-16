import pytest
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.ml.cli_mlops import mlops as cli
from click.testing import CliRunner
import os
import traceback


def click_success(result):
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0






def test_mlops_run(tmp_path):

    runner = CliRunner()
    # mock AWS credentials
    os.environ["AWS_ACCESS_KEY_ID"] = "dummy"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "dummy"
    os.environ["BUCKET_NAME"] = "dummy"
    os.environ["BUCKET_PREFIX"] = "dummy"
    
    # mock WANDB credentials
    os.environ["WANDB_PROJECT"] = "dummy"

    # mock CDD credentials
    os.environ["CDD_API_KEY"] = "dummy"
    os.environ["CDD_VAULT_NUMBER"] = "1"
    
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
    
    assert click_success(result)