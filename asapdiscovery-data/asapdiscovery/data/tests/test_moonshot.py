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
