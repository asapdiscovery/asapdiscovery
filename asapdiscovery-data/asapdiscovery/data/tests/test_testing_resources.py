import pytest
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.mark.parametrize("file", ["file1.txt", "file2.txt", "subdir/file3.txt"])
def test_get_file(file):
    path = fetch_test_file(file)
    assert path.exists()
    assert path.is_file()


@pytest.mark.parametrize("file", ["file5.txt", "file6.txt", "subdir/file7.txt"])
def test_get_fake_file(file):
    with pytest.raises(ValueError, match="Could not fetch test file"):
        path = fetch_test_file(file)
