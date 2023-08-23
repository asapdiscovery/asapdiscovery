import numpy as np

from typing import Literal

from pydantic import Field

from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.selectors import LigandSelectorBase
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.complex import Complex


class MCSLigandSelector(LigandSelectorBase):
    expander_type: Literal["PairwiseLigandSelector"] = "PairwiseLigandSelector"

    structure_based: bool = Field(
        False,
        description="Whether to use a structure-based search (True) or a more strict element-based search (False).",
    )

    def _select(
        self, ligands: list[Ligand], complexes: list[Complex], n_draw: int = 1
    ) -> list[tuple[Ligand, Complex]]:
        """
        Selects ligand and complex pairs based on maximum common substructure
        (MCS) search.

        Parameters
        ----------
        ligands : list[Ligand]
            List of ligands to search for
        complexes : list[Complex]
            List of complexes to search in
        n_draw : int, optional
            Draw top n_draw matched molecules

        """

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

            for i in range(n_draw):
                pairs.append(ligand, complexes_sorted[i])

        return pairs

    def provenance(self):
        return {"selector": self.dict(), "oechem": oechem.OEChemGetVersion()}
