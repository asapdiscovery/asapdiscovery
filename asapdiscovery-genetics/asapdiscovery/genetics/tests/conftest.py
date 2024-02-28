import pytest


@pytest.fixture(scope="session")
def example_fixture():
    return "example"