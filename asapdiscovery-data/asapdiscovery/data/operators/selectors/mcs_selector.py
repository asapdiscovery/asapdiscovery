import logging
import warnings
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
from dask import delayed
from pydantic import Field
from rdkit import Chem, rdBase
from rdkit.Chem import rdRascalMCES

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

        if self.approximate:
            warnings.warn(
                "Approximate MCS search is not guaranteed to find the maximum common substructure, see: https://docs.eyesopen.com/toolkits/python/oechemtk/patternmatch.html ",
                UserWarning,
            )

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


class RascalMCESSelector(SelectorBase):
    """
    Implementation of the Rascal Maximum Common Edge Subgraphs (MCES) algorithm
    from the following paper: https://eprints.whiterose.ac.uk/3568/1/willets3.pdf

    See this RDKit blog post for more information:
    https://greglandrum.github.io/rdkit-blog/posts/2023-11-08-introducingrascalmces.html

    The algorithm is implemented in RDKit as `rdRascalMCES.FindMCES` and is used
    to find the maximum common substructure between two molecules.

    Works pairwise between a ligand and a complex, and selects the complex with the
    highest similarity to the ligand.
    """

    selector_type: ClassVar[str] = "RascalMCESSelector"
    similarity_threshold: float = Field(
        0.7,
        description="Threshold for the similarity score, if the similarity is below this value, the RascalMCES algorithm will not attempt to find the MCS.",
    )

    def select(
        self,
        ligands: list[Ligand],
        complexes: list[Union[Complex, PreppedComplex]],
        use_dask: bool = False,
        dask_client=None,
        failure_mode: str = FailureMode.SKIP,
        **kwargs,
    ) -> list[Union[CompoundStructurePair, DockingInputPair]]:
        outputs = self._select(
            ligands=ligands,
            complexes=complexes,
            use_dask=use_dask,
            dask_client=dask_client,
            failure_mode=failure_mode,
            **kwargs,
        )
        return outputs

    def _select(
        self,
        ligands: list[Ligand],
        complexes: list[Union[Complex, PreppedComplex]],
        n_select: int = 1,
        use_dask: bool = False,
        dask_client=None,
        failure_mode: str = FailureMode.SKIP,
    ) -> list[Union[CompoundStructurePair, DockingInputPair]]:

        if not all(isinstance(c, ComplexBase) for c in complexes):
            raise ValueError("All complexes must be of type Complex, or PreppedComplex")

        if not all(isinstance(c, type(complexes[0])) for c in complexes):
            raise ValueError("All complexes must be of the same type")

        pair_cls = self._pair_type_from_complex(complexes[0])

        # clip n_select if it is larger than length of complexes to search from
        n_select = min(n_select, len(complexes))

        pairs = []

        for ligand in ligands:
            lsmiles = ligand.smiles
            similarities = []
            for complex in complexes:
                clsmiles = complex.ligand.smiles

                if use_dask:
                    similarity = delayed(self._single_pair_rascalMCES_similarity)(
                        lsmiles,
                        clsmiles,
                        similarity_threshold=self.similarity_threshold,
                    )
                else:
                    similarity = self._single_pair_rascalMCES_similarity(
                        lsmiles,
                        clsmiles,
                        similarity_threshold=self.similarity_threshold,
                    )

                similarities.append(similarity)

            if use_dask:
                similarities = actualise_dask_delayed_iterable(
                    similarities, dask_client=dask_client, errors=failure_mode
                )

            similarities = np.array(similarities)
            sort_idx = np.argsort(similarities)[::-1]  # sort in descending order
            complexes_sorted = np.asarray(complexes)[sort_idx]

            for i in range(n_select):
                pairs.append(pair_cls(ligand=ligand, complex=complexes_sorted[i]))

        return pairs

    @staticmethod
    def _single_pair_rascalMCES_similarity(
        lig_smiles: str, complex_lig_smiles: str, similarity_threshold: float = 0.7
    ) -> float:
        """
        Serializes the ligand and complex ligand SMILES strings into RDKit Mol objects
        then uses the rascalMCES algorithm to find the maximum common edge subgraph.

        The RDKit objects are instantiated inside this method to avoid pickling issues
        when using Dask delayed functions.

        The returned value is the the Johnson similarity score between the ligand and complex ligand.

        Parameters
        ----------
        lig_smiles : str
            SMILES string of the ligand
        complex_lig_smiles : str
            SMILES string of the complex ligand
        similarity_threshold : float
            Threshold for the similarity score, if the similarity is below this value, the RascalMCES algorithm will not attempt to find the MCS.

        Returns
        -------
        float
            Johnson similarity score between the ligand and complex ligand
        """
        lig_mol = Chem.MolFromSmiles(lig_smiles)
        complex_lig_mol = Chem.MolFromSmiles(complex_lig_smiles)
        opts = rdRascalMCES.RascalOptions()
        opts.returnEmptyMCES = True
        opts.singleLargestFrag = True
        opts.similarityThreshold = similarity_threshold

        try:
            mces = rdRascalMCES.FindMCES(lig_mol, complex_lig_mol, opts)
            similarity = mces[0].similarity
        except Exception as e:
            logger.error(f"Error in rascalMCES: {e}")
            similarity = 0.0

        return similarity

    def provenance(self):
        return {"selector": self.dict(), "rdkit": rdBase.rdkitVersion}
