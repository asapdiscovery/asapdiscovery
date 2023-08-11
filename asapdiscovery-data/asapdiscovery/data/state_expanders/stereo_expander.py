from typing import Literal

from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.openeye import oechem, oeomega
from asapdiscovery.data.state_expanders.state_expander import StateExpanderBase, StateExpansion, StateExpanderType

from pydantic import Field



class StereoExpander(StateExpanderBase):
    """
    Expand a molecule to stereoisomers
    """

    expander_type: Literal[StateExpanderType.STEREO] = StateExpanderType.STEREO
    stereo_expand_defined: bool = Field(False, description="Expand stereochemistry at defined centers as well as undefined centers")

    def _expand(self):
        
        flipperOpts = oeomega.OEFlipperOptions()
        flipperOpts.SetEnumSpecifiedStereo(self.stereo_expand_defined)

        expansions = []

        for ligand in self.input_ligands:
            expanded_states = []

            oemol = ligand.to_oemol()
            for enantiomer in oeomega.OEFlipper(oemol, flipperOpts):

                fmol = oechem.OEMol(enantiomer)
                # copy the ligand properties over to the new molecule, we may want to have more fine grained control over this
                # down the track.
                expanded_states.append(Ligand.from_oemol(fmol, **ligand.dict()))

            expansion = StateExpansion(parent=ligand, children=expanded_states, expander=self)
            expansions.append(expansion)

        return expansions
    
    

        

