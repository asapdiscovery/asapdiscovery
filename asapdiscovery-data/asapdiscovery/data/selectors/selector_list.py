"""
This collects all the selectors into a single enum
"""
from enum import Enum

from asapdiscovery.data.selectors.mcs_selector import MCSSelector
from asapdiscovery.data.selectors.pairwise_selector import (
    LeaveOneOutSelector,
    PairwiseSelector,
    SelfDockingSelector,
)

_ALL_SELECTORS = {LeaveOneOutSelector.selector_type: LeaveOneOutSelector}


class StructureSelector(Enum):
    """
    Enum of all selectors.
    """

    MCS = MCSSelector.selector_type
    PAIRWISE = PairwiseSelector.selector_type
    LEAVE_ONE_OUT = LeaveOneOutSelector.selector_type
    SELF_DOCKING = SelfDockingSelector.selector_type

    def to_selector_cls(self):
        """
        Returns the selector class.
        """
        return _ALL_SELECTORS[self.value]
