from itertools import product
from typing import Literal, Union

from asapdiscovery.data.schema_v2.complex import Complex, PreppedComplex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.selectors.selector import SelectorBase
from asapdiscovery.data.schema_v2.pairs import CompoundStructurePair, DockingInputPair


class PairwiseSelector(SelectorBase):
    """
    Selects ligand and complex pairs by enumerating all possible pairs.
    """

    expander_type: Literal["PairwiseSelector"] = "PairwiseSelector"

    def _select(
        self, ligands: list[Ligand], complexes: list[Union[Complex, PreppedComplex]]
    ) -> list[Union[CompoundStructurePair, DockingInputPair]]:
        return list(product(ligands, complexes))

    def provenance(self):
        return {"selector": self.dict()}
