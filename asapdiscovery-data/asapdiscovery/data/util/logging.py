import logging
import os
import sys
from typing import Optional, Union

from rich.console import Console
from rich.logging import RichHandler


class FileLogger:
    def __init__(
        self,
        logname: str,
        path: str,
        logfile: Optional[str] = None,
        level: Optional[Union[int, str]] = logging.DEBUG,
        format: Optional[
            str
        ] = "%(asctime)s | %(name)s | %(levelname)s | %(filename)s | %(funcName)s | %(message)s",
        stdout: Optional[bool] = False,
    ):
        self.name = logname
        self.logfile = logfile
        self.format = format
        self.level = level
        self.stdout = stdout

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)

        if self.logfile:
            self.handler = logging.FileHandler(
                os.path.join(path, self.logfile), mode="w"
            )
            self.handler.setLevel(self.level)
            self.formatter = logging.Formatter(self.format)
            self.handler.setFormatter(self.formatter)
            self.logger.addHandler(self.handler)

        if self.stdout:
            console = Console()
            rich_handler = RichHandler(console=console)
            self.logger.addHandler(rich_handler)

    def getLogger(self) -> logging.Logger:
        return self.logger

    def set_level(self, level: int) -> None:
        self.logger.setLevel(level)
        self.handler.setLevel(level)


"""
How to handle OpenEye error logging:
-----------------------------------

OpenEye has a global error stream that is set by the OEThrow.SetOutputStream() function.
This function takes an oeostream object as an argument.
It is not possible to set it from inside a function, class or method (??? why) at least when
I tried, (possibly some interaction with multiprocessing). We will need to use the following
snippet directly instead with a **UNIQUE FILENAME** if done in parallel. Note that you cannot use
the same filename as for another log file, because the interleave of the two streams doesn't
seem to work nicely and the log will be difficult to read and possibly missing information.


    errfs = oechem.oeofstream(os.path.join(out_dir, f"openeye_{logname}-log.txt"))
    oechem.OEThrow.SetOutputStream(errfs)

"""


class HiddenPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
