from pathlib import Path

import pandas
import pytest
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.services.fragalysis.fragalysis_reader import FragalysisFactory
from asapdiscovery.data.testing.test_resources import fetch_test_file
from pydantic.v1 import ValidationError


@pytest.fixture(scope="session")
def all_mpro_fns():
    return [
        "metadata.csv",
        "aligned/Mpro-x11041_0A/Mpro-x11041_0A_bound.pdb",
        "aligned/Mpro-x1425_0A/Mpro-x1425_0A_bound.pdb",
        "aligned/Mpro-x11894_0A/Mpro-x11894_0A_bound.pdb",
        "aligned/Mpro-x1002_0A/Mpro-x1002_0A_bound.pdb",
        "aligned/Mpro-x10155_0A/Mpro-x10155_0A_bound.pdb",
        "aligned/Mpro-x0354_0A/Mpro-x0354_0A_bound.pdb",
        "aligned/Mpro-x11271_0A/Mpro-x11271_0A_bound.pdb",
        "aligned/Mpro-x1101_1A/Mpro-x1101_1A_bound.pdb",
        "aligned/Mpro-x1187_0A/Mpro-x1187_0A_bound.pdb",
        "aligned/Mpro-x10338_0A/Mpro-x10338_0A_bound.pdb",
    ]


@pytest.fixture(scope="session")
def mpro_frag_dir(all_mpro_fns):
    all_paths = [fetch_test_file(f"frag_factory_test/{fn}") for fn in all_mpro_fns]
    return all_paths[0].parent, all_paths


@pytest.fixture
def mpro_frag_dir_only_metadata(mpro_frag_dir, tmp_path):
    (tmp_path / "metadata.csv").symlink_to(mpro_frag_dir[1][0])
    return tmp_path


@pytest.fixture
def mpro_frag_dir_missing_one_pdb(tmp_path, all_mpro_fns, mpro_frag_dir):
    # Make all aligned/ subdirectories
    for fn in all_mpro_fns[1:]:
        (tmp_path / fn).parent.mkdir(parents=True)

    parent_dir, all_paths = mpro_frag_dir
    # Link all files except last one
    for fn, fn_target in zip(all_mpro_fns[:-1], all_paths[:-1]):
        (tmp_path / fn).symlink_to(fn_target)

    return tmp_path


@pytest.fixture
def mpro_frag_compound_mapping(mpro_frag_dir):
    parent_dir, all_paths = mpro_frag_dir
    df = pandas.read_csv(all_paths[0], index_col=0)

    return dict(zip(df["crystal_name"], df["alternate_name"]))


def test_manual_creation(mpro_frag_dir, mpro_frag_compound_mapping):
    parent_dir, all_paths = mpro_frag_dir
    _ = [
        Complex.from_pdb(
            p,
            target_kwargs={"target_name": p.parts[-2]},
            ligand_kwargs={"compound_name": mpro_frag_compound_mapping[p.parts[-2]]},
        )
        for p in all_paths[1:]
    ]
    ff = FragalysisFactory(parent_dir=parent_dir)
    complexes = ff.load()
    assert len(complexes) == 10


@pytest.mark.parametrize("use_dask", [True, False])
def test_creation_from_dir(mpro_frag_dir, use_dask):
    parent_dir, all_paths = mpro_frag_dir
    ff = FragalysisFactory.from_dir(parent_dir)
    complexes = ff.load(use_dask=use_dask)
    assert len(complexes) == 10


def test_validation_fails_nonexistent(tmp_path):
    with pytest.raises(ValidationError, match="Given parent_dir does not exist."):
        _ = FragalysisFactory(parent_dir=(tmp_path / "nonexistent"))


def test_validation_fails_not_dir(tmp_path):
    p = tmp_path / "not_a_directory"
    p.touch()
    with pytest.raises(ValidationError, match="Given parent_dir is not a directory."):
        _ = FragalysisFactory(parent_dir=p)


def test_creation_fails_without_metadata(tmp_path):
    with pytest.raises(FileNotFoundError, match="file found in parent_dir."):
        _ = FragalysisFactory.from_dir(tmp_path, metadata_csv_name="wrong_file.csv")


def test_creation_fails_with_empty_metadata(tmp_path):
    p = tmp_path / "metadata.csv"
    aligned_dir = tmp_path / "aligned"
    aligned_dir.mkdir()
    p.touch()
    with p.open("w") as fp:
        fp.write("a,b,c")
    with pytest.raises(ValueError, match="file is empty."):
        ff = FragalysisFactory.from_dir(tmp_path)
        ff.load()


def test_creation_fails_without_proper_cols(tmp_path):
    p = tmp_path / "metadata.csv"
    p.touch()
    aligned_dir = tmp_path / "aligned"
    aligned_dir.mkdir()
    with p.open("w") as fp:
        fp.write("a,b,c\n")
        fp.write("1,2,3\n")
    with pytest.raises(
        ValueError,
        match=(
            "metadata.csv file must contain a crystal name column and a "
            "compound name column."
        ),
    ):
        ff = FragalysisFactory.from_dir(tmp_path)
        ff.load()

    with pytest.raises(
        ValueError,
        match=(
            "metadata.csv file must contain a crystal name column and a "
            "compound name column."
        ),
    ):
        ff = FragalysisFactory.from_dir(tmp_path, xtal_col="a")
        ff.load()

    with pytest.raises(
        ValueError,
        match=(
            "metadata.csv file must contain a crystal name column and a "
            "compound name column."
        ),
    ):
        ff = FragalysisFactory.from_dir(tmp_path, compound_col="a")
        ff.load()


def test_creation_fails_without_aligned(mpro_frag_dir_only_metadata):
    with pytest.raises(
        FileNotFoundError, match="No aligned/ directory found in parent_dir."
    ):
        _ = FragalysisFactory.from_dir(mpro_frag_dir_only_metadata)


def test_creation_fails_with_empty_aligned(mpro_frag_dir_only_metadata):
    (mpro_frag_dir_only_metadata / "aligned/test1").mkdir(parents=True)
    with pytest.raises(
        ValueError,
        match="No aligned directories found with entries in metadata.csv.",
    ):
        ff = FragalysisFactory.from_dir(mpro_frag_dir_only_metadata)
        ff.load()


def test_creation_fails_with_missing_file_fail_missing_true(
    all_mpro_fns, mpro_frag_dir_missing_one_pdb
):
    with pytest.raises(
        FileNotFoundError,
        match=f"No PDB file found for {Path(all_mpro_fns[-1]).parts[-2]}.",
    ):
        ff = FragalysisFactory.from_dir(
            mpro_frag_dir_missing_one_pdb, fail_missing=True
        )
        ff.load()


def test_creation_succeeds_with_missing_file_fail_missing_false(
    mpro_frag_dir_missing_one_pdb,
):
    _ = FragalysisFactory.from_dir(mpro_frag_dir_missing_one_pdb)


def test_ff_equal(mpro_frag_dir):
    parent_dir, _ = mpro_frag_dir

    ff1 = FragalysisFactory.from_dir(parent_dir)
    ff2 = FragalysisFactory.from_dir(parent_dir)

    assert ff1 == ff2


def test_ff_dict_roundtrip(mpro_frag_dir):
    parent_dir, _ = mpro_frag_dir

    ff1 = FragalysisFactory.from_dir(parent_dir)
    ff2 = FragalysisFactory.parse_obj(ff1.dict())

    assert ff1 == ff2
