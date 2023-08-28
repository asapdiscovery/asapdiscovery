from typing import Literal, Union

import numpy as np
from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.schema_v2.complex import Complex, ComplexBase, PreppedComplex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.pairs import CompoundStructurePair, DockingInputPair
from asapdiscovery.data.selectors.selector import SelectorBase
from pydantic import Field


class MCSSelector(SelectorBase):
    """
    Selects ligand and complex pairs based on a maximum common substructure
    (MCS) search.
    """

    expander_type: Literal["PairwiseSelector"] = "PairwiseSelector"

    structure_based: bool = Field(
        False,
        description="Whether to use a structure-based search (True) or a more strict element-based search (False).",
    )

    def _select(
        self,
        ligands: list[Ligand],
        complexes: list[Union[Complex, PreppedComplex]],
        n_select: int = 1,
    ) -> list[Union[CompoundStructurePair, DockingInputPair]]:
        """
        Selects ligand and complex pairs based on maximum common substructure
        (MCS) search.

        Parameters
        ----------
        ligands : list[Ligand]
            List of ligands to search for
        complexes : list[Union[Complex, PreppedComplex]]]
            List of complexes to search in
        n_select : int, optional
            Draw top n_select matched molecules for each ligand (default: 1) this means that the
            number of pairs returned is n_select * len(ligands)

        Returns
        -------
        list[tuple[Ligand, Complex]]
            List of ligand and complex pairs
        """

        if not all(isinstance(c, ComplexBase) for c in complexes):
            raise ValueError("All complexes must be of the same type")

        pair_cls = self._pair_type_from_complex(complexes[0])

        # clip n_select if it is larger than length of complexes to search from
        n_select = min(n_select, len(complexes))

        if self.structure_based:
            """
            For structure based matching
            Options for atom matching:
            * Aromaticity
            * HvyDegree - # heavy atoms bonded to
            * RingMember
            Options for bond matching:
            * Aromaticity
            * BondOrder
            * RingMember
            """
            atomexpr = (
                oechem.OEExprOpts_Aromaticity
                | oechem.OEExprOpts_HvyDegree
                | oechem.OEExprOpts_RingMember
            )
            bondexpr = (
                oechem.OEExprOpts_Aromaticity
                | oechem.OEExprOpts_BondOrder
                | oechem.OEExprOpts_RingMember
            )
        else:
            """
            For atom based matching
            Options for atom matching (predefined AutomorphAtoms):
            * AtomicNumber
            * Aromaticity
            * RingMember
            * HvyDegree - # heavy atoms bonded to
            Options for bond matching:
            * Aromaticity
            * BondOrder
            * RingMember
            """
            atomexpr = oechem.OEExprOpts_AutomorphAtoms
            bondexpr = (
                oechem.OEExprOpts_Aromaticity
                | oechem.OEExprOpts_BondOrder
                | oechem.OEExprOpts_RingMember
            )

        # Set up the search pattern and MCS objects
        pairs = []

        for ligand in ligands:
            pattern_query = oechem.OEQMol(ligand.to_oemol())
            pattern_query.BuildExpressions(atomexpr, bondexpr)
            mcss = oechem.OEMCSSearch(pattern_query)
            mcss.SetMCSFunc(oechem.OEMCSMaxAtomsCompleteCycles())

            sort_args = []
            for complex in complexes:
                complex_mol = complex.ligand.to_oemol()
                # MCS search
                mcs = next(iter(mcss.Match(complex_mol, True)))
                sort_args.append((mcs.NumBonds(), mcs.NumAtoms()))

            sort_args = np.asarray(sort_args)
            sort_idx = np.lexsort(-sort_args.T)

            complexes_sorted = np.asarray(complexes)[sort_idx]

            for i in range(n_select):
                pairs.append(pair_cls(ligand=ligand, complex=complexes_sorted[i]))

        return pairs

    def provenance(self):
        return {"selector": self.dict(), "oechem": oechem.OEChemGetVersion()}
