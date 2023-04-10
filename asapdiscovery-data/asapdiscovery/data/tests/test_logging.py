import logging
import pytest

from asapdiscovery.data.logging import FileLogger


def run_func(name):
    logger = logging.getLogger(f"internal_logger.{name}")
    logger.info("Default test")


@pytest.fixture()
def file_logger():
    file_logger = FileLogger("top_logger", ".")
    logger = file_logger.getLogger()
    logger.info("Top level test")
    return file_logger


@pytest.mark.parametrize("name", ["test1", "test2"])
def test_toplevel(file_logger, name: str):
    with open(file_logger.logfile) as f:
        fread = f.read()
        assert "Top level test" in fread


@pytest.mark.parametrize("name", ["test1", "test2"])
def test_set_as_default(file_logger, name: str):
    file_logger.set_as_default()
    assert file_logger


@pytest.mark.parametrize("name", ["test1", "test2"])
def test_internal_func(file_logger, name: str):
    file_logger = FileLogger(f"internal_logger.{name}", ".")
    logger = file_logger.getLogger()
    logger.info("Running run_func")
    run_func(name)
    with open(file_logger.logfile) as f:
        fread = f.read()
        assert "Top level test" not in fread
        assert "Default test" in fread
