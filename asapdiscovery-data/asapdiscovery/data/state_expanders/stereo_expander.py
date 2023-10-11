from typing import Literal

from asapdiscovery.data.openeye import oechem, oeomega
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.state_expanders.state_expander import StateExpanderBase
from pydantic import Field


class StereoExpander(StateExpanderBase):
    """
    Expand a molecule to stereoisomers
    """

    expander_type: Literal["StereoExpander"] = "StereoExpander"
    stereo_expand_defined: bool = Field(
        False,
        description="Expand stereochemistry at defined centers `True` as well as undefined centers",
    )

    def _provenance(self) -> dict[str, str]:
        return {
            "oechem": oechem.OEChemGetVersion(),
            "omega": oeomega.OEOmegaGetVersion(),
        }

    def _check_stereo_matches(
        self, ref_ligand: oechem.OEMol, enantiomer: oechem.OEMol
    ) -> bool:
        """
        Make sure that the stereo tags for all defined centers and bonds match between the two molecules.
        Parameters
        ----------
        ref_ligand
        enantiomer

        Returns
        -------

        """
        for ref_atom, enantiomer_atom in zip(
            ref_ligand.GetAtoms(), enantiomer.GetAtoms()
        ):
            if ref_atom.IsChiral() and ref_atom.HasStereoSpecified():
                if oechem.OEPerceiveCIPStereo(
                    ref_ligand, ref_atom
                ) != oechem.OEPerceiveCIPStereo(enantiomer, enantiomer_atom):
                    return False
        return True

    def _expand(self, ligands: list[Ligand]) -> list[Ligand]:
        """
        Expand the stereoisomers of the input molecules.

        Note:
        The input molecules are not included in the outputs.

        Parameters
        ----------
        ligands: The list of ligands who's states should be expanded.

        Returns
        -------
            A list of expanded ligand states.

        """
        flipperOpts = oeomega.OEFlipperOptions()
        # Work around openeye only expanding the first center when many are not specified
        flipperOpts.SetEnumSpecifiedStereo(True)
        provenance = self.provenance()

        expanded_states = []

        for parent_ligand in ligands:
            oemol = parent_ligand.to_oemol()
            for enantiomer in oeomega.OEFlipper(oemol, flipperOpts):
                fmol = oechem.OEMol(enantiomer)
                enantiomer_ligand = Ligand.from_oemol(fmol, **parent_ligand.dict())
                # copy the ligand properties over to the new molecule, we may want to have more fine grained control over this
                # down the track.
                # filter out molecules which expand the wrong centers
                if not self.stereo_expand_defined and not self._check_stereo_matches(
                    ref_ligand=oemol, enantiomer=fmol
                ):
                    continue
                enantiomer_ligand = Ligand.from_oemol(fmol, **parent_ligand.dict())
                enantiomer_ligand.set_expansion(parent_ligand, provenance=provenance)
                expanded_states.append(enantiomer_ligand)

        return expanded_states
