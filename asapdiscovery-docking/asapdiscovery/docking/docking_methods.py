"""
This collects the docking methods together in a single enum
"""
from enum import Enum

from asapdiscovery.docking.docking_v2 import POSITDocker


class DockingMethod(Enum):
    """
    Enum of all docking methods.
    """

    POSIT = POSITDocker
