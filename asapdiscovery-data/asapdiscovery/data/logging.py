import logging
import os

from asapdiscovery.data.openeye import oechem


class FileLogger:
    def __init__(self, logname: str, path: str, logfile: str = None):
        self.name = logname
        self.logfile = logfile
        if self.logfile is None:
            self.logfile = self.logname + "-log.txt"

        self.logger = logging.getLogger(self.logfile)
        self.logger.setLevel(logging.DEBUG)
        self.handler = logging.FileHandler(os.path.join(path, self.logfile), mode="w")
        self.handler.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

    def getLogger(self):
        return self.logger


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
