"""
Tests for downloading and processing Moonshot data from CDD.
"""
import os

import pytest
from asapdiscovery.data.moonshot import (
    ALL_SMI_SEARCH,
    CDD_URL,
    MOONSHOT_VAULT,
    NONCOVALENT_SMI_SEARCH,
    NONCOVALENT_W_DATES_SEARCH,
    download_molecules,
    download_url,
)
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.fixture(scope="session")
def dl_dir(tmp_path_factory):
    """
    Temporary download directory that will persist for the duration of the testing
    session. This will let us keep downloaded files to test file caching.
    """
    return tmp_path_factory.mktemp("cache", numbered=False)


@pytest.fixture
def cdd_header():
    try:
        token_fn = os.environ["CDDTOKEN"]
    except KeyError:
        # All tests need to be able to download files, so stop early if there's no API key
        pytest.exit("CDDTOKEN environment variable not set.", 1)
    try:
        token = "".join(open(token_fn).readlines()).strip()
    except OSError:
        pytest.exit("Failed to read CDDTOKEN file.", 1)

    return {"X-CDD-token": token}


@pytest.fixture
def filter_df_files():
    """
    Fetch all possible combination of output filtering CSV files. Filenames are built by
    all filter kwargs that are True, put together in the following order:
     * achiral
     * racemic
     * enantiopure
     * semiquant
    """
    from itertools import product

    fn_labels = ["achiral", "racemic", "enantiopure", "semiquant"]
    out_fn_dict = {}
    # achiral, racemic, enant, semiquant
    for flags in product([True, False], repeat=len(fn_labels)):
        out_fn = "_".join([label for label, flag in zip(fn_labels, flags) if flag])
        out_fn = fetch_test_file(
            "test_filter" + (f"_{out_fn}" if out_fn else "") + "_out.csv"
        )
        out_fn_dict[flags] = out_fn

    in_fn = fetch_test_file("test_filter_in.csv")

    return in_fn, out_fn_dict


@pytest.fixture
def parse_df_files():
    """
    Fetch all possible combination of output parsed CSV files. Filenames are built by
    bool representations of keep_best_per_mol and cp_values.
    """
    from itertools import product

    fn_labels = ["best", "cheng"]
    out_fn_dict = {}
    for flags in product([True, False], repeat=len(fn_labels)):
        out_fn = "_".join([label for label, flag in zip(fn_labels, flags) if flag])
        out_fn = fetch_test_file(
            "test_parse" + (f"_{out_fn}" if out_fn else "") + "_out.csv"
        )
        out_fn_dict[flags] = out_fn

    in_fn = fetch_test_file("test_parse_in.csv")

    return in_fn, out_fn_dict


@pytest.mark.parametrize(
    "search", [ALL_SMI_SEARCH, NONCOVALENT_SMI_SEARCH, NONCOVALENT_W_DATES_SEARCH]
)
def test_fetch(cdd_header, search):
    """
    Test fetching all saved search.
    """
    url = f"{CDD_URL}/{MOONSHOT_VAULT}/searches/{search}"
    response = download_url(url, cdd_header, vault=MOONSHOT_VAULT)
    assert response.ok


@pytest.mark.parametrize(
    "search,fn",
    [
        (ALL_SMI_SEARCH, "all_smi.csv"),
        (NONCOVALENT_SMI_SEARCH, "noncov_smi.csv"),
        (NONCOVALENT_W_DATES_SEARCH, "noncov_smi_dates.csv"),
    ],
)
def test_save(cdd_header, search, dl_dir, fn):
    """
    Test saving after fetching.
    """
    url = f"{CDD_URL}/{MOONSHOT_VAULT}/searches/{search}"
    response = download_url(url, cdd_header, vault=MOONSHOT_VAULT)
    content = response.content.decode()

    cache_fn = dl_dir / fn
    with cache_fn.open("w") as outfile:
        outfile.write(content)


@pytest.mark.parametrize(
    "fn",
    ["all_smi.csv", "noncov_smi.csv", "noncov_smi_dates.csv"],
)
def test_saved_files_exist(dl_dir, fn):
    assert (dl_dir / fn).exists()


@pytest.mark.parametrize(
    "fn",
    ["all_smi.csv", "noncov_smi.csv", "noncov_smi_dates.csv"],
)
def test_saved_files_can_be_loaded(dl_dir, fn):
    import pandas

    df = pandas.read_csv(dl_dir / fn)
    print(df.shape, df.columns, flush=True)


@pytest.mark.parametrize("retain_achiral", [True, False])
@pytest.mark.parametrize("retain_racemic", [True, False])
@pytest.mark.parametrize("retain_enantiopure", [True, False])
@pytest.mark.parametrize("retain_semiquantitative_data", [True, False])
def test_filter_df(
    retain_achiral,
    retain_racemic,
    retain_enantiopure,
    retain_semiquantitative_data,
    filter_df_files,
):
    import pandas
    from asapdiscovery.data.utils import filter_molecules_dataframe

    in_fn, all_out_fns = filter_df_files
    flags = (
        retain_achiral,
        retain_racemic,
        retain_enantiopure,
        retain_semiquantitative_data,
    )
    out_fn = all_out_fns[flags]

    in_df = pandas.read_csv(in_fn)
    out_df = pandas.read_csv(out_fn)

    in_df_filtered = filter_molecules_dataframe(
        in_df,
        retain_achiral=retain_achiral,
        retain_racemic=retain_racemic,
        retain_enantiopure=retain_enantiopure,
        retain_semiquantitative_data=retain_semiquantitative_data,
    )

    assert in_df_filtered.shape[0] == out_df.shape[0]
    assert (
        in_df_filtered["name"].values == out_df["Canonical PostEra ID"].values
    ).all()


@pytest.mark.parametrize("keep_best", [True, False])
@pytest.mark.parametrize("cp_values", [None, [0.375, 9.5]])
def test_parse_fluorescence(keep_best, cp_values, parse_df_files):
    print(keep_best, cp_values, flush=True)
    import pandas
    from asapdiscovery.data.utils import parse_fluorescence_data_cdd

    in_fn, all_out_fns = parse_df_files
    flags = (keep_best, bool(cp_values))
    out_fn = all_out_fns[flags]

    in_df = pandas.read_csv(in_fn)
    out_df = pandas.read_csv(out_fn)

    in_df_parsed = parse_fluorescence_data_cdd(
        in_df, keep_best_per_mol=keep_best, cp_values=cp_values
    )

    # Check that range values were assigned correctly
    assert (in_df_parsed["pIC50_range"] == out_df["pIC50_range"]).all()

    # Columns with float vals to compare
    float_check_cols = [
        "IC50 (M)",
        "IC50_stderr (M)",
        "IC50_95ci_lower (M)",
        "IC50_95ci_upper (M)",
        "pIC50",
        "pIC50_stderr",
        "pIC50_95ci_lower",
        "pIC50_95ci_upper",
        "exp_binding_affinity_kcal_mol",
        "exp_binding_affinity_kcal_mol_95ci_lower",
        "exp_binding_affinity_kcal_mol_95ci_upper",
        "exp_binding_affinity_kcal_mol_stderr",
    ]
    for c in float_check_cols:
        assert _check_parsed_vals(in_df_parsed[c], out_df[c])


def _check_parsed_vals(col1, col2):
    """
    Helper function for test_parse_fluorescence to compare two numerical columns,
    appropriately handling checking for null/non-numeric values.

    Parameters
    ----------
    col1, col2 : pandas.Series
        The two DF columns to compare

    Returns
    -------
    bool
        If the two cols are equivalent
    """
    import numpy as np

    # Indices with NaN values so they can be compared appropriately
    nan_idx1 = col1.isna()
    nan_idx2 = col2.isna()

    # Chceck that nans are the same
    if not (nan_idx1 == nan_idx2).all():
        return False

    # Check that all non-nan values are close
    return np.isclose(col1[~nan_idx1], col2[~nan_idx2]).all()
