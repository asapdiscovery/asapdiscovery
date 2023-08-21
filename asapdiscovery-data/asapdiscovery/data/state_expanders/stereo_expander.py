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
        description="Expand stereochemistry at defined centers as well as undefined centers",
    )

    def provenance(self) -> dict[str, str]:
        return {
            "expander": self.expander_type,
            "oechem": oechem.OEChemGetVersion(),
            "omega": oeomega.OEOmegaGetVersion(),
        }

    def _expand(self, ligands: list[Ligand]) -> list[Ligand]:
        flipperOpts = oeomega.OEFlipperOptions()
        flipperOpts.SetEnumSpecifiedStereo(self.stereo_expand_defined)

        expanded_states = []

        for parent_ligand in ligands:
            oemol = parent_ligand.to_oemol()
            parent_ligand.make_parent_tag(provenance=self.provenance())
            for enantiomer in oeomega.OEFlipper(oemol, flipperOpts):
                fmol = oechem.OEMol(enantiomer)
                # copy the ligand properties over to the new molecule, we may want to have more fine grained control over this
                # down the track.
                enantiomer_ligand = Ligand.from_oemol(fmol, **parent_ligand.dict())
                enantiomer_ligand.set_parent(
                    parent_ligand, provenance=self.provenance()
                )
                expanded_states.append(enantiomer_ligand)

        return expanded_states
