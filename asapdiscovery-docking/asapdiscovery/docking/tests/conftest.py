from pathlib import Path
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--local_path",
        type=str,
        default=None,
        help="If provided, use this path to output files for tests",
    )


@pytest.fixture(scope="session")
def local_path(request):
    return request.config.getoption("--local_path")


# This needs to have a scope of session so that a new tmp file is not created for each test
@pytest.fixture(scope="session")
def output_dir(tmp_path_factory, local_path):
    if not type(local_path) == str:
        return tmp_path_factory.mktemp("test_prep")
    else:
        local_path = Path(local_path)
        local_path.mkdir(exist_ok=True)
        assert local_path.exists()
        return local_path
