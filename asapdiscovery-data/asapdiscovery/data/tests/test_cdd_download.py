"""
Tests for downloading and processing Moonshot data from CDD.
"""

import os

import pandas
import pytest
import requests_mock
from asapdiscovery.data.services.cdd.cdd_download import (
    CDD_URL,
    MOONSHOT_ALL_SMI_SEARCH,
    MOONSHOT_NONCOVALENT_SMI_SEARCH,
    MOONSHOT_NONCOVALENT_W_DATES_SEARCH,
    download_molecules,
    download_url,
)
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.data.util.utils import (
    filter_molecules_dataframe,
    parse_fluorescence_data_cdd,
)
from numpy.testing import assert_allclose

# Columns added by filter_molecules_dataframe
FILTER_ADDED_COLS = ["name", "smiles", "achiral", "racemic", "enantiopure", "semiquant"]

# Columns added by parse_fluorescence_data_cdd
PARSE_ADDED_COLS = [
    "IC50 (M)",
    "IC50_stderr (M)",
    "IC50_95ci_lower (M)",
    "IC50_95ci_upper (M)",
    "pIC50",
    "pIC50_stderr",
    "pIC50_range",
    "pIC50_95ci_lower",
    "pIC50_95ci_upper",
    "exp_binding_affinity_kcal_mol",
    "exp_binding_affinity_kcal_mol_stderr",
    "exp_binding_affinity_kcal_mol_95ci_lower",
    "exp_binding_affinity_kcal_mol_95ci_upper",
]

# skip tests if no CDD token
# pytestmark = pytest.mark.skipif(not os.getenv("CDDTOKEN"), reason="No CDD token")


@pytest.fixture(scope="session")
def cdd_header():
    try:
        token = os.environ["CDDTOKEN"]
    except KeyError:
        # All tests need to be able to download files, so stop early if there's no API key
        pytest.exit("CDDTOKEN environment variable not set.", 1)

    return {"X-CDD-token": token}


@pytest.fixture(scope="session")
def moonshot_vault():
    try:
        vault = os.environ["MOONSHOT_CDD_VAULT_NUMBER"]
    except KeyError:
        # All tests need to be able to download files, so stop early if there's no API key
        pytest.exit("MOONSHOT_CDD_VAULT_NUMBER environment variable not set.", 1)

    return vault


@pytest.fixture(scope="session")
def moonshot_saved_searches(tmp_path_factory, cdd_header, moonshot_vault):
    # Hashes of the saved search downloads for quick comparison
    # Unless we get really unlucky, files downloaded within a few minutes of each other
    #  should contain the same entries
    from hashlib import sha256

    def dl_and_hash(search, fn_out):
        url = f"{CDD_URL}/{moonshot_vault}/searches/{search}"
        response = download_url(url, cdd_header, vault=moonshot_vault)

        with fn_out.open("w") as fp:
            fp.write(response.content.decode())

        return sha256(response.content).hexdigest()

    dl_dir = tmp_path_factory.mktemp("cache", numbered=False)
    hash_dict = {
        search: dl_and_hash(search, dl_dir / search)
        for search in [
            MOONSHOT_ALL_SMI_SEARCH,
            MOONSHOT_NONCOVALENT_SMI_SEARCH,
            MOONSHOT_NONCOVALENT_W_DATES_SEARCH,
        ]
    }

    return dl_dir, hash_dict


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


@pytest.fixture
def cdd_col_headers():
    return {
        MOONSHOT_ALL_SMI_SEARCH: [
            "Molecule Name",
            "Batch Created Date",
            "Batch Updated Date",
            "Canonical PostEra ID",
            "stereochem comments",
            "shipment_SMILES",
            "suspected_SMILES",
            "why_suspected_SMILES",
            "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 (µM)",
            "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Lower) (µM)",
            "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Upper) (µM)",
            "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Hill slope",
            "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Curve class",
        ],
        MOONSHOT_NONCOVALENT_SMI_SEARCH: [
            "Molecule Name",
            "CDD Number",
            "Canonical PostEra ID",
            "Scaffold",
            "stereochem comments",
            "shipment_SMILES",
            "suspected_SMILES",
            "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 (µM)",
            "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Lower) (µM)",
            "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Upper) (µM)",
            "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Hill slope",
            "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Curve class",
            "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Avg pIC50",
        ],
        MOONSHOT_NONCOVALENT_W_DATES_SEARCH: [
            "Molecule Name",
            "CDD Number",
            "Batch Created Date",
            "Batch Updated Date",
            "Canonical PostEra ID",
            "Scaffold",
            "stereochem comments",
            "shipment_SMILES",
            "suspected_SMILES",
            "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 (µM)",
            "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Lower) (µM)",
            "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Upper) (µM)",
            "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Hill slope",
            "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Curve class",
            "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Avg pIC50",
        ],
    }


@pytest.mark.xfail(reason="CDD download can be flaky")
@pytest.mark.parametrize(
    "search",
    [
        MOONSHOT_ALL_SMI_SEARCH,
        MOONSHOT_NONCOVALENT_SMI_SEARCH,
        MOONSHOT_NONCOVALENT_W_DATES_SEARCH,
    ],
)
def test_fetch(cdd_header, moonshot_vault, search, cdd_col_headers):
    """
    Test fetching all saved search.
    """
    url = f"{CDD_URL}/{moonshot_vault}/searches/{search}"
    response = download_url(url, cdd_header, vault=moonshot_vault)
    assert response.ok

    # Check that the correct data was downloaded
    #  different number of molecules depdending on when the download occurred, so just
    #  check the column header
    content = response.content.decode()
    lines = content.split("\n")
    assert lines[0] == ",".join(cdd_col_headers[search])


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
        assert_allclose(
            in_df_parsed[c].values,
            out_df[c].values,
            rtol=1e-05,
            atol=1e-08,
            equal_nan=True,
            err_msg=f"{c} cols not equal",
        )


@pytest.mark.xfail
@pytest.mark.parametrize(
    "search",
    [
        MOONSHOT_ALL_SMI_SEARCH,
        MOONSHOT_NONCOVALENT_SMI_SEARCH,
        MOONSHOT_NONCOVALENT_W_DATES_SEARCH,
    ],
)
def test_download_molecules(
    cdd_header,
    moonshot_vault,
    search,
    cdd_col_headers,
    moonshot_saved_searches,
    tmp_path,
):
    from hashlib import sha256

    import pandas

    # Download and check
    fn_out = tmp_path / "out.csv"
    fn_cache = tmp_path / "cache.csv"
    df = download_molecules(
        cdd_header,
        vault=moonshot_vault,
        search=search,
        fn_out=fn_out,
        fn_cache=fn_cache,
    )

    # Extra columns will be added
    target_cols = cdd_col_headers[search] + FILTER_ADDED_COLS + PARSE_ADDED_COLS
    assert sorted(df.columns.tolist()) == sorted(target_cols)

    df_loaded = pandas.read_csv(fn_out)
    assert sorted(df_loaded.columns.tolist()) == sorted(target_cols)

    assert (
        sha256(fn_cache.open("rb").read()).hexdigest()
        == moonshot_saved_searches[1][search]
    )


@pytest.mark.xfail
@pytest.mark.parametrize(
    "search",
    [
        MOONSHOT_ALL_SMI_SEARCH,
        MOONSHOT_NONCOVALENT_SMI_SEARCH,
        MOONSHOT_NONCOVALENT_W_DATES_SEARCH,
    ],
)
def test_download_molecules_cache(
    cdd_header, moonshot_vault, search, cdd_col_headers, moonshot_saved_searches
):
    # First download file
    saved_fn = moonshot_saved_searches[0] / search

    # Search will only be run if loading from cache fails
    df = download_molecules(
        cdd_header,
        vault=moonshot_vault,
        search="non_existent_search",
        fn_cache=saved_fn,
    )
    target_cols = cdd_col_headers[search] + FILTER_ADDED_COLS + PARSE_ADDED_COLS
    assert sorted(df.columns.tolist()) == sorted(target_cols)


def test_cdd_api_get_molecules_exclusive_args(mocked_cdd_api):
    """Make sure an error is raised if we pass mutually exclusive args."""

    with pytest.raises(ValueError):
        _ = mocked_cdd_api.get_molecules(
            smiles="CCO", names=["ASAP-Ethanol"], compound_ids=[1, 2]
        )


@pytest.mark.parametrize(
    "search, expected_result",
    [
        pytest.param({"smiles": "CO"}, None, id="smiles missing"),
        pytest.param(
            {"smiles": "CCO"},
            [{"name": "ethanol", "smiles": "CCO", "id": 1}],
            id="Smiles",
        ),
        pytest.param(
            {"names": ["ASAP-Ethanol"]},
            [{"name": "ethanol", "smiles": "CCO", "id": 1}],
            id="Names",
        ),
        pytest.param(
            {"compound_ids": [1]},
            [{"name": "ethanol", "smiles": "CCO", "id": 1}],
            id="Compound ids",
        ),
    ],
)
def test_cdd_api_get_molecules(mocked_cdd_api, search, expected_result):
    """Test searching via the api using different molecule identifiers."""

    def get_mols(request, *args):
        "mock the molecule get request"
        data = request.json()
        if "structure" in data and data["structure"] != "CCO":
            return {"count": 0, "objects": []}
        elif "structure" in data:
            return {
                "count": 1,
                "objects": [{"name": "ethanol", "smiles": "CCO", "id": 1}],
            }
        else:
            # if not we do an async request
            return {"id": "1"}

    with requests_mock.Mocker() as m:
        m.get(mocked_cdd_api.api_url + "molecules/", json=get_mols)
        m.get(
            mocked_cdd_api.api_url + "exports/1",
            json={
                "count": 1,
                "objects": [{"name": "ethanol", "smiles": "CCO", "id": 1}],
            },
        )
        result = mocked_cdd_api.get_molecules(**search)
        assert result == expected_result


def test_cdd_api_get_molecules_missing(mocked_cdd_api):
    """Test getting molecules by ID and handling the case with missing molecules"""

    def get_mols(request, *args):
        "mock the molecule request to give an error for missing mols"
        data = request.json()
        if 2 in data["molecules"]:
            return {"error": "2, and 3 not found", "code": 404}
        elif data["molecules"] == [1]:
            return {"id": "1"}

    with requests_mock.Mocker() as m:
        m.get(mocked_cdd_api.api_url + "molecules/", json=get_mols)
        m.get(
            mocked_cdd_api.api_url + "exports/1",
            json={
                "count": 1,
                "objects": [{"name": "ethanol", "smiles": "CCO", "id": 1}],
            },
        )
        result = mocked_cdd_api.get_molecules(compound_ids=[1, 2, 3])
        assert result == [{"name": "ethanol", "smiles": "CCO", "id": 1}]


@pytest.mark.parametrize(
    "protocol_names",
    [pytest.param(None, id="No names"), pytest.param(["p1", "p2"], id="Names")],
)
def test_cdd_api_get_protocol(mocked_cdd_api, protocol_names):
    """Test pulling down protocol data."""

    with requests_mock.Mocker() as m:
        m.get(mocked_cdd_api.api_url + "protocols", json={"objects": [{"id": 1}]})
        result = mocked_cdd_api.get_protocols(protocol_names=protocol_names)
        assert result == [{"id": 1}]


def test_cdd_api_readout_rows(mocked_cdd_api):
    """Test pulling down readout data using the api"""

    with requests_mock.Mocker() as m:
        m.get(mocked_cdd_api.api_url + "readout_rows", json={"id": 2})
        m.get(
            mocked_cdd_api.api_url + "exports/2",
            json={"count": 2, "objects": [{"id": 2}, {"id": 3}]},
        )
        # do a query with two results
        result = mocked_cdd_api.get_readout_rows(
            protocol=2, types=["batch_run_aggregate_row"], molecule_ids=[1, 2]
        )
        assert result == [{"id": 2}, {"id": 3}]


def test_cdd_api_get_ic50(mocked_cdd_api):
    """Test a full run of finding ic50 data for a given protocol.
    Get ready to mock everything!
    """
    assay_name = "My_fancy_assay"
    mock_protocol_response = {
        "objects": [
            {
                "id": 1,
                "readout_definitions": [
                    {"name": "IC50", "id": 500},
                    {"name": "IC50 CI (Lower)", "id": 501},
                    {"name": "IC50 CI (Upper)", "id": 502},
                    {"name": "Curve class", "id": 503},
                ],
            }
        ]
    }
    mock_readout_response = {
        "count": 1,
        "objects": [
            {
                "id": 1,
                "molecule": 1,
                "readouts": {
                    "500": {"value": 0.03},
                    "501": {"value": 0.028},
                    "502": {"value": 0.031},
                    "503": {"value": 1.1},
                },
                "modified_at": "2021-01-01T00:00:00Z",
            }
        ],
    }
    mock_molecule_response = {
        "count": 1,
        "objects": [
            {
                "id": 1,
                "smiles": "CCO",
                "inchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
                "inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "name": "ASAP-Ethanol",
                "cxsmiles": "CCO",
            }
        ],
    }
    with requests_mock.Mocker() as m:
        # mock the required protocols
        m.get(mocked_cdd_api.api_url + "protocols", json=mock_protocol_response)
        # mock the return of the async request
        m.get(mocked_cdd_api.api_url + "readout_rows", json={"id": 100})
        # mock the export request for the readout rows
        m.get(mocked_cdd_api.api_url + "exports/100", json=mock_readout_response)
        # mock the molecule async request
        m.get(mocked_cdd_api.api_url + "molecules/", json={"id": 101})
        m.get(mocked_cdd_api.api_url + "exports/101", json=mock_molecule_response)
        ic50_df = mocked_cdd_api.get_ic50_data(protocol_name=assay_name)
        # check the values were collected correctly
        ethanol_data = ic50_df.iloc[0]
        assert ethanol_data[f"{assay_name}: IC50 (µM)"] == 0.03
        assert ethanol_data[f"{assay_name}: IC50 CI (Lower) (µM)"] == 0.028
        assert ethanol_data[f"{assay_name}: IC50 CI (Upper) (µM)"] == 0.031
        assert ethanol_data[f"{assay_name}: Curve class"] == 1.1
        assert ethanol_data["modified_at"] == "2021-01-01T00:00:00Z"
        # check the molecule identifiers
        mol_data = mock_molecule_response["objects"][0]
        assert mol_data["smiles"] == ethanol_data["Smiles"]
        assert mol_data["inchi"] == ethanol_data["Inchi"]
        assert mol_data["inchi_key"] == ethanol_data["Inchi Key"]
        assert mol_data["name"] == ethanol_data["Molecule Name"]
