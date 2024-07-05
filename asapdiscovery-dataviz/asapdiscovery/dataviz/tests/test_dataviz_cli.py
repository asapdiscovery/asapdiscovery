import traceback

import pytest
from asapdiscovery.dataviz.cli import visualization
from click.testing import CliRunner


def click_success(result):
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("color_method", ["subpockets", "fitness"])
@pytest.mark.parametrize("align", [True, False])
def test_viz_cli(tmp_path, pose, protein, use_dask, color_method, align):
    runner = CliRunner()
    args = [
        "pose-html",
        "--colour-method",
        color_method,
        "--target",
        "SARS-CoV-2-Mpro",
        "--ligands",
        pose,
        "--pdb-file",
        protein,
        "--output-dir",
        tmp_path,
        "--loglevel",
        "INFO",
    ]
    if align:
        args.append("--align")
    if use_dask:
        args.append("--use-dask")
    result = runner.invoke(visualization, args)
    assert click_success(result)


@pytest.mark.parametrize("pymol_debug", [False, True])
def test_gif_cli(tmp_path, traj, top, pymol_debug):
    runner = CliRunner()
    args = [
        "traj-gif",
        "--target",
        "SARS-CoV-2-Mpro",
        "--traj",
        traj,
        "--top",
        top,
        "--output-dir",
        tmp_path,
    ]
    if pymol_debug:
        args.append("--pymol-debug")
    result = runner.invoke(visualization, args)
    assert click_success(result)
