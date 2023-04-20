import logging
import os
from typing import Optional


class FileLogger:
    def __init__(
        self,
        logname: str,
        path: str,
        logfile: Optional[str] = None,
        level: Optional[int] = logging.DEBUG,
        format: Optional[
            str
        ] = "%(asctime)s | %(name)s | %(levelname)s | %(filename)s | %(funcName)s | %(message)s",
    ):
        self.name = logname
        self.logfile = logfile
        self.format = format
        self.level = level
        if self.logfile is None:
            self.logfile = self.name + "-log.txt"

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        self.handler = logging.FileHandler(os.path.join(path, self.logfile), mode="w")
        self.handler.setLevel(self.level)
        self.formatter = logging.Formatter(self.format)
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

    def getLogger(self) -> logging.Logger:
        return self.logger

    def set_as_default(self) -> None:
        logging.basicConfig(filename=self.logfile, level=self.level, format=self.format)


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
