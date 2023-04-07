import logging

from asapdiscovery.data.logging import FileLogger


def run_func(name):
    logger = logging.getLogger(f"internal_logger.{name}")
    logger.info("Default test")


def internal_func(name: str):
    file_logger = FileLogger(f"internal_logger.{name}", ".")
    logger = file_logger.getLogger()
    logger.info("Running run_func")
    run_func(name)
    with open(file_logger.logfile) as f:
        assert "Default test" in f.read()


def main():
    # logging.basicConfig(filename="different_test.txt", level=logging.DEBUG)
    file_logger = FileLogger("top_logger", ".")
    logger = file_logger.getLogger()
    logger.info("Top level test")
    internal_func("test1")
    internal_func("test2")


if __name__ == "__main__":
    main()
