import traceback

import pytest
from asapdiscovery.cli.cli import cli
from click.testing import CliRunner


def click_success(result):
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0


def test_toplevel_runnable():
    runner = CliRunner()
    args = ["--help"]
    result = runner.invoke(cli, args)
    assert click_success(result)


@pytest.mark.parametrize(
    "subcommand",
    [
        "protein-prep",
        "docking",
        "alchemy",
        "genetics",
        "ml",
        "visualization",
        "simulation",
    ],
)
def test_subcommand_runnable(subcommand):
    runner = CliRunner()
    args = [subcommand, "--help"]
    result = runner.invoke(cli, args)
    assert click_success(result)
