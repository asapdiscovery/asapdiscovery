import pytest
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.mark.parametrize(
    "file", ["internal_testing/file1.txt", "Mpro_combined_labeled.sdf"]
)
def test_get_file(file):
    path = fetch_test_file(file)
    assert path.exists()
    assert path.is_file()


def test_get_file_from_list():
    paths = fetch_test_file(["internal_testing/file1.txt", "Mpro_combined_labeled.sdf"])
    for path in paths:
        assert path.exists()
        assert path.is_file()


@pytest.mark.parametrize("file", ["file5.txt", "file6.txt", "subdir/file7.txt"])
def test_get_fake_file(file):
    with pytest.raises(ValueError, match="Could not fetch test file"):
        _ = fetch_test_file(file)


# below is the recommended way to use the fetch_test_file functionality in a test
# to avoid thrashing


@pytest.fixture(scope="session")
def file1():
    return fetch_test_file("internal_testing/file1.txt")


def test_use_file_fixture(file1):
    assert file1.exists()
    assert file1.is_file()
