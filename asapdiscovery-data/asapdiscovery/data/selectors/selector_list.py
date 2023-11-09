"""
This collects all the selectors into a single enum
"""
from enum import Enum

from asapdiscovery.data.selectors.mcs_selector import MCSSelector
from asapdiscovery.data.selectors.pairwise_selector import (
    PairwiseSelector,
    LeaveOneOutSelector,
    SelfDockingSelector,
)


class StructureSelector(Enum):
    """
    Enum of all selectors.
    """

    MCS = MCSSelector
    PAIRWISE = PairwiseSelector
    LEAVE_ONE_OUT = LeaveOneOutSelector
    SELF_DOCKING = SelfDockingSelector
