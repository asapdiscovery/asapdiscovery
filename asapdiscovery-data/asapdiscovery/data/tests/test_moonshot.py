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

    Note that these files don't exist yet.
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


@pytest.mark.parametrize("retain_achiral", [True, False])
@pytest.mark.parametrize("retain_racemic", [True, False])
@pytest.mark.parametrize("retain_enantiopure", [True, False])
@pytest.mark.parametrize("retain_semiquantitative_data", [True, False])
def test_filter_df(
    retain_achiral, retain_racemic, retain_enantiopure, retain_semiquantitative_data
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

    in_df = pandas.from_csv(in_fn)
    out_df = pandas.from_csv(out_fn)

    in_df_filtered = filter_molecules_dataframe(
        in_df,
        retain_achiral=retain_achiral,
        retain_racemic=retain_racemic,
        retain_enantiopure=retain_enantiopure,
        retain_semiquantitative_data=retain_semiquantitative_data,
    )

    assert in_df_filtered.shape == out_df.shape
    assert (in_df_filtered.index == out_df.index).all()
