from itertools import product
from typing import Literal, Union

from asapdiscovery.data.schema_v2.complex import Complex, PreppedComplex, ComplexBase
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
        if not all(isinstance(c, ComplexBase) for c in complexes):
            raise ValueError("All complexes must be of the same type")

        pair_cls = self._pair_type_from_complex(complexes[0])

        pairs = []
        for lig, complex in product(ligands, complexes):
            pairs.append(pair_cls(complex=complex, ligand=lig))

        return pairs

    def provenance(self):
        return {"selector": self.dict()}
