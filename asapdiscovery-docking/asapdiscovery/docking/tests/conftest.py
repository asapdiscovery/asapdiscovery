from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def local_path(request):
    try:
        return request.config.getoption("--local_path")
    except ValueError:
        return None


# This needs to have a scope of session so that a new tmp file is not created for each test
@pytest.fixture(scope="session")
def output_dir(tmp_path_factory, local_path):
    if type(local_path) is not str:
        return tmp_path_factory.mktemp("test_prep")
    else:
        local_path = Path(local_path)
        local_path.mkdir(exist_ok=True)
        assert local_path.exists()
        return local_path
