import pytest
from asapdiscovery.data.readers.structure_dir import StructureDirFactory
from asapdiscovery.data.testing.test_resources import fetch_test_file


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


@pytest.mark.parametrize("use_dask", [True, False])
def test_structure_dir(use_dask, structure_dir):
    struct_dir, _ = structure_dir
    factory = StructureDirFactory.from_dir(struct_dir)
    complexes = factory.load(use_dask=use_dask)
    assert len(complexes) == 2


def test_custom_glob(structure_dir):
    struct_dir, _ = structure_dir
    factory = StructureDirFactory.from_dir(struct_dir)
    factory.glob = "*x1002*.pdb"
    complexes = factory.load()
    assert len(complexes) == 1
