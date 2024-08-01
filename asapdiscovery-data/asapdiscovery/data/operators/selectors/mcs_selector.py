import logging
from typing import ClassVar, Union

import numpy as np
from asapdiscovery.data.backend.openeye import oechem
from asapdiscovery.data.operators.selectors.selector import SelectorBase
from asapdiscovery.data.schema.complex import Complex, ComplexBase, PreppedComplex
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.schema.pairs import CompoundStructurePair
from asapdiscovery.data.util.dask_utils import (
    FailureMode,
    actualise_dask_delayed_iterable,
)
from asapdiscovery.docking.docking import DockingInputPair  # TODO: move to backend
from pydantic import Field

logger = logging.getLogger(__name__)


def sort_by_mcs(
    reference_ligand: Ligand,
    target_ligands: list[Ligand],
    structure_matching: bool = False,
) -> np.array:
    """
    Get the sorted order of the target ligands by the MCS overlap with the reference ligand.

    Args:
        reference_ligand: The ligand the targets should be matched to.
        target_ligands: The list of target ligands which should be ordered.
        structure_matching: If structure-based matching `True` should be used or element-based `False`.

    Returns:
        An array of the target ligand indices ordered by MCS overlap.
    """

    # generate the matching expressions
    if structure_matching is True:
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

    # use the ref mol as the pattern
    pattern_query = oechem.OEQMol(reference_ligand.to_oemol())
    pattern_query.BuildExpressions(atomexpr, bondexpr)
    mcss = oechem.OEMCSSearch(pattern_query)
    mcss.SetMCSFunc(oechem.OEMCSMaxAtomsCompleteCycles())

    mcs_matches = []

    for ligand in target_ligands:
        # MCS search on each ligand
        try:
            mcs = next(iter(mcss.Match(ligand.to_oemol(), True)))
            mcs_matches.append((mcs.NumBonds(), mcs.NumAtoms()))
        except StopIteration:  # no match found
            mcs_matches.append((0, 0))

    # sort by the size of the MCS match
    sort_args = np.asarray(mcs_matches)
    sort_idx = np.lexsort(-sort_args.T)
    return sort_idx


class MCSSelector(SelectorBase):
    """
    Selects ligand and complex pairs based on a maximum common substructure
    (MCS) search.
    """

    selector_type: ClassVar[str] = "MCSSelector"

    structure_based: bool = Field(
        False,
        description="Whether to use a structure-based search (True) or a more strict element-based search (False).",
    )
    approximate: bool = Field(
        False,
        description="Whether to use an approximate MCS search (True) or an exact MCS search (False).",
    )

    def select(
        self,
        ligands: list[Ligand],
        complexes: list[Union[Complex, PreppedComplex]],
        **kwargs,
    ) -> list[Union[CompoundStructurePair, DockingInputPair]]:
        outputs = self._select(ligands=ligands, complexes=complexes, **kwargs)
        return outputs

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
            raise ValueError("All complexes must be of type Complex, or PreppedComplex")

        if not all(isinstance(c, type(complexes[0])) for c in complexes):
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

            # If only one complex is available, skip the MCS search
            if len(complexes) == 1:
                pairs.append(pair_cls(ligand=ligand, complex=complexes[0]))
                continue

            pattern_query = oechem.OEQMol(ligand.to_oemol())
            pattern_query.BuildExpressions(atomexpr, bondexpr)
            if self.approximate:
                mcs_stype = oechem.OEMCSType_Approximate
            else:
                mcs_stype = oechem.OEMCSType_Exhaustive
            mcss = oechem.OEMCSSearch(pattern_query, True, mcs_stype)
            mcss.SetMCSFunc(oechem.OEMCSMaxAtomsCompleteCycles())

            sort_args = []
            for complex in complexes:
                complex_mol = complex.ligand.to_oemol()
                # MCS search
                try:
                    mcs = next(iter(mcss.Match(complex_mol, True)))
                    sort_args.append((mcs.NumBonds(), mcs.NumAtoms()))
                except StopIteration:  # no match found
                    sort_args.append((0, 0))
            sort_args = np.asarray(sort_args)
            sort_idx = np.lexsort(-sort_args.T)
            complexes_sorted = np.asarray(complexes)[sort_idx]

            for i in range(n_select):
                pairs.append(pair_cls(ligand=ligand, complex=complexes_sorted[i]))

        return pairs

    def provenance(self):
        return {"selector": self.dict(), "oechem": oechem.OEChemGetVersion()}


def _mcs_inner_row(mcss, complexes, n_select, ligand, pair_cls):
    """
    Do one dimension of the NxM MCS search, ie for a single ligand
    check all complexes for MCS overlap.
    """
    for complex in complexes:
        complex_mol = complex.ligand.to_oemol()
        # MCS search
        sort_args = []
        try:
            mcs = next(iter(mcss.Match(complex_mol, True)))
            sort_args.append((mcs.NumBonds(), mcs.NumAtoms()))
        except StopIteration:
            sort_args.append((0, 0))

    sort_args = np.asarray(sort_args)
    sort_idx = np.lexsort(-sort_args.T)
    complexes_sorted = np.asarray(complexes)[sort_idx]

    pairs = []
    for i in range(n_select):
        pairs.append(pair_cls(ligand=ligand, complex=complexes_sorted[i]))
    return pairs
