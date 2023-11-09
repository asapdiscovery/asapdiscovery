from itertools import product
from typing import Literal, Union

from asapdiscovery.data.schema_v2.complex import Complex, ComplexBase, PreppedComplex
from asapdiscovery.data.schema_v2.ligand import Ligand, compound_names_unique
from asapdiscovery.data.schema_v2.pairs import CompoundStructurePair
from asapdiscovery.data.selectors.selector import SelectorBase
from asapdiscovery.docking.docking_v2 import DockingInputPair


class PairwiseSelector(SelectorBase):
    """
    Selects ligand and complex pairs by enumerating all possible pairs.
    """

    expander_type: Literal["PairwiseSelector"] = "PairwiseSelector"

    def _select(
        self, ligands: list[Ligand], complexes: list[Union[Complex, PreppedComplex]]
    ) -> list[Union[CompoundStructurePair, DockingInputPair]]:
        if not all(isinstance(c, ComplexBase) for c in complexes):
            raise ValueError("All complexes must be of type Complex, or PreppedComplex")

        if not all(isinstance(c, type(complexes[0])) for c in complexes):
            raise ValueError("All complexes must be of the same type")

        pair_cls = self._pair_type_from_complex(complexes[0])

        pairs = []
        for lig, complex in product(ligands, complexes):
            pairs.append(pair_cls(complex=complex, ligand=lig))

        return pairs

    def provenance(self):
        return {"selector": self.dict()}


class LeaveOneOutSelector(SelectorBase):
    """
    Selects ligand and complex pairs by enumerating all possible pairs except the self-docked pair
    """

    expander_type: Literal["LeaveOneOutSelector"] = "LeaveOneOutSelector"

    def _select(
        self, ligands: list[Ligand], complexes: list[Union[Complex, PreppedComplex]]
    ) -> list[Union[CompoundStructurePair, DockingInputPair]]:
        if not all(isinstance(c, ComplexBase) for c in complexes):
            raise ValueError("All complexes must be of type Complex, or PreppedComplex")

        if not all(isinstance(c, type(complexes[0])) for c in complexes):
            raise ValueError("All complexes must be of the same type")

        pair_cls = self._pair_type_from_complex(complexes[0])

        pairs = []
        for lig, complex in product(ligands, complexes):
            if not lig.inchi == complex.ligand.inchi:
                print("Lig 1:", lig)
                print("Lig 2:", complex.ligand)
                pairs.append(pair_cls(complex=complex, ligand=lig))

        return pairs

    def provenance(self):
        return {"selector": self.dict()}


class SelfDockingSelector(SelectorBase):
    """
    Selects ligand and complex pairs only including the self-docked pair
    """

    expander_type: Literal["SelfDockingSelector"] = "SelfDockingSelector"

    def _select(
        self, ligands: list[Ligand], complexes: list[Union[Complex, PreppedComplex]]
    ) -> list[Union[CompoundStructurePair, DockingInputPair]]:
        if not all(isinstance(c, ComplexBase) for c in complexes):
            raise ValueError("All complexes must be of type Complex, or PreppedComplex")

        if not all(isinstance(c, type(complexes[0])) for c in complexes):
            raise ValueError("All complexes must be of the same type")

        pair_cls = self._pair_type_from_complex(complexes[0])

        pairs = []
        for lig, complex in product(ligands, complexes):
            if lig.inchi == complex.ligand.inchi:
                pairs.append(pair_cls(complex=complex, ligand=lig))

        return pairs

    def provenance(self):
        return {"selector": self.dict()}
