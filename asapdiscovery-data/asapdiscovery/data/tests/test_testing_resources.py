import pytest
from asapdiscovery.data.testing.test_resources import fetch_test_file


@pytest.mark.parametrize("file", ["file1.txt", "file2.txt", "subdir/file3.txt"])
def test_get_file(file):
    path = fetch_test_file(file)
    assert path.exists()
    assert path.is_file()
