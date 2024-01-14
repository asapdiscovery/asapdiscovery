"""
This collects all the selectors into a single enum
"""
from enum import Enum

from asapdiscovery.data.selectors.mcs_selector import MCSSelector
from asapdiscovery.data.selectors.pairwise_selector import (
    LeaveOneOutSelector,
    LeaveSimilarOutSelector,
    PairwiseSelector,
    SelfDockingSelector,
)

_ALL_SELECTORS = {
    LeaveOneOutSelector.selector_type: LeaveOneOutSelector,
    LeaveSimilarOutSelector.selector_type: LeaveSimilarOutSelector,
    PairwiseSelector.selector_type: PairwiseSelector,
    SelfDockingSelector.selector_type: SelfDockingSelector,
    MCSSelector.selector_type: MCSSelector,
}


class StructureSelector(str, Enum):
    """
    Enum of all selectors.
    """

    MCS = MCSSelector.selector_type
    PAIRWISE = PairwiseSelector.selector_type
    LEAVE_ONE_OUT = LeaveOneOutSelector.selector_type
    LEAVE_SIMILAR_OUT = LeaveSimilarOutSelector.selector_type
    SELF_DOCKING = SelfDockingSelector.selector_type

    @property
    def selector_cls(self):
        """
        Returns the selector class.
        """
        return _ALL_SELECTORS[self.value]

    @classmethod
    def get_values(cls):
        """
        Returns all the values of the enum.
        """
        return [selector.value for selector in cls]
