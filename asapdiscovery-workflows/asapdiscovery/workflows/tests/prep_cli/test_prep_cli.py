import os
import traceback

import pytest
from asapdiscovery.data.metadata.resources import master_structures
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.workflows.prep_workflows.cli import protein_prep as cli
from click.testing import CliRunner


def click_success(result):
    if result.exit_code != 0:  # -no-cov-  (only occurs on test error)
        print(result.output)
        traceback.print_tb(result.exc_info[2])
        print(result.exc_info[0], result.exc_info[1])
    return result.exit_code == 0


def _get_target_struct_pairs():
    # for ZIKA_NS2B_NS3pro, we use a different structure as the canonical one is a fragment screen
    rvals = [
        (k, v)
        for k, v in master_structures.items()
        if k not in ["ZIKV-NS2B-NS3pro", "SARS-CoV-2-Mac1-monomer"]
    ]
    rvals.append(
        ("ZIKV-NS2B-NS3pro", fetch_test_file("zikv_nsb2_nsb3_literature_structure.pdb"))
    )
    return rvals


@pytest.mark.parametrize("target, structure", _get_target_struct_pairs())
@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
@pytest.mark.skipif(os.getenv("SKIP_EXPENSIVE_TESTS"), reason="Expensive tests skipped")
def test_project_support_prep(
    target,
    structure,
    tmp_path,
):
    runner = CliRunner()
    args = [
        "--target",
        target,
        "--pdb-file",
        structure,
        "--output-dir",
        tmp_path,
    ]
    result = runner.invoke(cli, args)
    assert click_success(result)
