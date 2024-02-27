import logging
from itertools import product
from typing import ClassVar, Union

from asapdiscovery.data.operators.selectors.selector import SelectorBase
from asapdiscovery.data.schema.complex import Complex, ComplexBase, PreppedComplex
from asapdiscovery.data.schema.ligand import ChemicalRelationship, Ligand
from asapdiscovery.data.schema.pairs import CompoundStructurePair

logger = logging.getLogger(__name__)


class PairwiseSelector(SelectorBase):
    """
    Selects ligand and complex pairs by enumerating all possible pairs.
    """

    selector_type: ClassVar[str] = "PairwiseSelector"

    def _select(
        self, ligands: list[Ligand], complexes: list[Union[Complex, PreppedComplex]]
    ) -> list[CompoundStructurePair]:
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

    selector_type: ClassVar[str] = "LeaveOneOutSelector"

    def _select(
        self, ligands: list[Ligand], complexes: list[Union[Complex, PreppedComplex]]
    ) -> list[CompoundStructurePair]:
        if not all(isinstance(c, ComplexBase) for c in complexes):
            raise ValueError("All complexes must be of type Complex, or PreppedComplex")

        if not all(isinstance(c, type(complexes[0])) for c in complexes):
            raise ValueError("All complexes must be of the same type")

        pair_cls = self._pair_type_from_complex(complexes[0])

        pairs = []
        for lig, complex in product(ligands, complexes):
            # Need to compare chemical identity instead of compound ID
            if not lig.inchi == complex.ligand.inchi:
                pairs.append(pair_cls(complex=complex, ligand=lig))

        return pairs

    def provenance(self):
        return {"selector": self.dict()}


class LeaveSimilarOutSelector(SelectorBase):
    """
    Selects ligand and complex pairs by enumerating all possible pairs except any that are similar (including not just
    identical ligands but also stereoisomers, protonation states, and tautomers).
    """

    selector_type: ClassVar[str] = "LeaveSimilarOutSelector"

    def _select(
        self, ligands: list[Ligand], complexes: list[Union[Complex, PreppedComplex]]
    ) -> list[CompoundStructurePair]:
        if not all(isinstance(c, ComplexBase) for c in complexes):
            raise ValueError("All complexes must be of type Complex, or PreppedComplex")

        if not all(isinstance(c, type(complexes[0])) for c in complexes):
            raise ValueError("All complexes must be of the same type")

        pair_cls = self._pair_type_from_complex(complexes[0])

        pairs = []
        for lig, complex in product(ligands, complexes):
            # Need to compare chemical identity instead of compound ID
            if (
                lig.get_chemical_relationship(complex.ligand)
                not in ChemicalRelationship.IDENTICAL
                | ChemicalRelationship.STEREOISOMER
                | ChemicalRelationship.TAUTOMER
                | ChemicalRelationship.PROTONATION_STATE_ISOMER
            ):
                pairs.append(pair_cls(complex=complex, ligand=lig))

        return pairs

    def provenance(self):
        return {"selector": self.dict()}


class SelfDockingSelector(SelectorBase):
    """
    Selects ligand and complex pairs only including the self-docked pair
    """

    selector_type: ClassVar[str] = "SelfDockingSelector"

    def _select(
        self, ligands: list[Ligand], complexes: list[Union[Complex, PreppedComplex]]
    ) -> list[CompoundStructurePair]:
        if not all(isinstance(c, ComplexBase) for c in complexes):
            raise ValueError("All complexes must be of type Complex, or PreppedComplex")

        if not all(isinstance(c, type(complexes[0])) for c in complexes):
            raise ValueError("All complexes must be of the same type")

        pair_cls = self._pair_type_from_complex(complexes[0])

        pairs = []
        for lig, complex in product(ligands, complexes):
            # Need to compare chemical identity instead of compound ID
            if lig.inchi == complex.ligand.inchi:
                pairs.append(pair_cls(complex=complex, ligand=lig))

        return pairs

    def provenance(self):
        return {"selector": self.dict()}
