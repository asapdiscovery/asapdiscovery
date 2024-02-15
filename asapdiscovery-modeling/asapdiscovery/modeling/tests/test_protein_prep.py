import pytest
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.sequence import seqres_by_target
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.modeling.protein_prep import ProteinPrepper


@pytest.fixture
def loop_db():
    return fetch_test_file("fragalysis-mpro_spruce.loop_db")


@pytest.fixture(scope="session")
def cmplx():
    return Complex.from_pdb(
        fetch_test_file("structure_dir/Mpro-x0354_0A_bound.pdb"),
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test2"},
    )


@pytest.fixture(scope="session")
def prep_complex():
    return Complex.from_pdb(
        fetch_test_file("SARS2_Mac1A-A1013.pdb"),
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test2"},
    )


@pytest.fixture(scope="session")
def all_structure_dir_fns():
    return [
        "structure_dir/Mpro-x0354_0A_bound.pdb",
        "structure_dir/Mpro-x1002_0A_bound.pdb",
    ]


@pytest.fixture(scope="session")
def structure_dir(all_structure_dir_fns):
    all_paths = [fetch_test_file(f) for f in all_structure_dir_fns]
    return all_paths[0].parent, all_paths


@pytest.fixture(scope="session")
def du_cache_files():
    return ["du_cache/Mpro-x0354_0A_bound.oedu", "du_cache/Mpro-x1002_0A_bound.oedu"]


@pytest.fixture(scope="session")
def du_cache(du_cache_files):
    all_paths = [fetch_test_file(f) for f in du_cache_files]
    return all_paths[0].parent, all_paths


@pytest.fixture(scope="session")
def json_cache():
    """A mock json cache of prepared proteins"""
    return fetch_test_file("protein_json_cache/Mpro-x0354_0A_bound.json")


@pytest.mark.parametrize("use_dask", [True, False])
def test_protein_prep(prep_complex, use_dask, loop_db):
    target = TargetTags["SARS-CoV-2-Mac1"]
    prepper = ProteinPrepper(loop_db=loop_db, seqres_yaml=seqres_by_target(target))
    pcs = prepper.prep([prep_complex], use_dask=use_dask)
    assert len(pcs) == 1
    assert pcs[0].target.target_name == "test"
    assert pcs[0].ligand.compound_name == "test2"


def test_cache_load(json_cache):
    """Test loading cached PreppedComplex files."""

    cached_complexs = ProteinPrepper.load_cache(cache_dir=json_cache.parent)
    assert len(cached_complexs) == 1
    assert (
        cached_complexs[0].hash
        == "9e2ea19d1a175314647dacb9d878138a80b8443cff5faf56031bf4af61179a0a+GIIIJZOPGUFGBF-QXYFZJGFNA-O"
    )
