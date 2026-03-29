import json
import math

import pandas
import pytest
from click.testing import CliRunner

from asapdiscovery.data.cli.cli import data as cli
from asapdiscovery.data.testing.test_resources import fetch_test_file


def _normalize_nan(obj):
    """Recursively replace NaN with None to handle pydantic v1/v2 serialization differences."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _normalize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_nan(v) for v in obj]
    return obj


@pytest.fixture(scope="session")
def cdd_to_schema_files():
    in_fn = fetch_test_file("test_cdd_to_schema_in.csv")

    out_json_fn = fetch_test_file("test_cdd_to_schema_out.json")
    out_csv_fn = fetch_test_file("test_cdd_to_schema_out.csv")

    return in_fn, out_csv_fn, out_json_fn


def test_cdd_to_schema(cdd_to_schema_files, tmp_path):
    in_fn, out_csv_fn, out_json_fn = cdd_to_schema_files

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "cdd-to-schema",
            "--in-file",
            in_fn,
            "--out-json",
            tmp_path / "test_cdd_to_schema_out.json",
            "--out-csv",
            tmp_path / "test_cdd_to_schema_out.csv",
        ],
    )
    assert result.exit_code == 0

    # Make sure files exist
    test_out_json_fn = tmp_path / "test_cdd_to_schema_out.json"
    test_out_csv_fn = tmp_path / "test_cdd_to_schema_out.csv"
    assert test_out_json_fn.exists()
    assert test_out_csv_fn.exists()

    # Check files are right
    df_check = pandas.read_csv(out_csv_fn, index_col=0)
    df_test = pandas.read_csv(test_out_csv_fn, index_col=0)
    assert df_test.equals(df_check)

    json_check = _normalize_nan(json.loads(out_json_fn.read_text()))
    json_test = _normalize_nan(json.loads(test_out_json_fn.read_text()))

    json_check = sorted(json_check, key=lambda d: d.get("compound_id"))
    json_test = sorted(json_test, key=lambda d: d.get("compound_id"))
    assert all([d1 == d2 for d1, d2 in zip(json_check, json_test)])
