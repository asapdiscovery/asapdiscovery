"""
This collects all the selectors into a single enum
"""
from enum import Enum
from mcs_selector import MCSSelector
from pairwise_selector import PairwiseSelector


class StructureSelector(Enum):
    """
    Enum of all selectors.
    """

    MCS = MCSSelector
    PAIRWISE = PairwiseSelector
