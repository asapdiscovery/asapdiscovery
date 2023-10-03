import logging

import pytest
from asapdiscovery.data.logging import FileLogger


def run_func(name):
    logger = logging.getLogger(name)
    logger.info("Default test")


def test_toplevel(tmp_path):
    file_logger = FileLogger("top_logger", tmp_path, level=logging.INFO)
    logger = file_logger.getLogger()
    logger.info("Top level test")
    with open(tmp_path / file_logger.logfile) as f:
        fread = f.read()
        assert "Top level test" in fread


def test_internal_func(tmp_path):
    file_logger = FileLogger("top_logger", tmp_path, level=logging.INFO)
    logger = file_logger.getLogger()
    logger.info("Top level test")
    run_func(file_logger.name)
    with open(tmp_path / file_logger.logfile) as f:
        fread = f.read()
        assert "Top level test" in fread
        assert "Default test" in fread
