from typing import Literal

from pydantic import Field

from asapdiscovery.data.openeye import oechem, oeomega
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.state_expanders.state_expander import (
    StateExpanderBase,
    StateExpansion,
)


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
            "oechem": oechem.OEChemGetVersion(),
            "omega": oeomega.OEOmegaGetVersion(),
        }

    def _expand(self, ligands: list[Ligand]) -> list[StateExpansion]:
        flipperOpts = oeomega.OEFlipperOptions()
        flipperOpts.SetEnumSpecifiedStereo(self.stereo_expand_defined)

        expansions = []

        for ligand in ligands:
            expanded_states = []

            oemol = ligand.to_oemol()
            for enantiomer in oeomega.OEFlipper(oemol, flipperOpts):
                fmol = oechem.OEMol(enantiomer)
                # copy the ligand properties over to the new molecule, we may want to have more fine grained control over this
                # down the track.
                expanded_states.append(Ligand.from_oemol(fmol, **ligand.dict()))

            expansion = StateExpansion(
                parent=ligand,
                children=expanded_states,
                expander=self.dict(),
                provenance=self.provenance(),
            )
            expansions.append(expansion)

        return expansions
