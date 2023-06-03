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
