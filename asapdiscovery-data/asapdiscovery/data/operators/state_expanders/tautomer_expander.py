from typing import Literal

from asapdiscovery.data.backend.openeye import clear_SD_data, oechem, oequacpac
from asapdiscovery.data.operators.state_expanders.state_expander import (
    StateExpanderBase,
)
from asapdiscovery.data.schema.ligand import Ligand
from pydantic.v1 import Field


class TautomerExpander(StateExpanderBase):
    """
    Expand a molecule to reasonable tautomers using OpenEye.

    Note:
        The input molecule is also returned.
    """

    expander_type: Literal["TautomerExpander"] = "TautomerExpander"
    tautomer_save_stereo: bool = Field(
        False, description="Preserve stereochemistry in tautomers"
    )
    tautomer_carbon_hybridization: bool = Field(
        True, description="Allow carbon hybridization changes in tautomers"
    )
    pka_norm: bool = Field(
        True,
        description="If true the ionization state of each tautomer will be assigned to a predominate state at pH~7.4.",
    )

    def _provenance(self) -> dict[str, str]:
        return {
            "oechem": oechem.OEChemGetVersion(),
            "quacpac": oequacpac.OEQuacPacGetVersion(),
        }

    def _expand(self, ligands: list[Ligand]) -> list[Ligand]:
        tautomer_opts = oequacpac.OETautomerOptions()
        tautomer_opts.SetSaveStereo(self.tautomer_save_stereo)
        tautomer_opts.SetCarbonHybridization(self.tautomer_carbon_hybridization)

        expanded_states = []
        provenance = self.provenance()

        for parent_ligand in ligands:
            # need to clear the SD data otherwise the provenance will break
            oemol = clear_SD_data(parent_ligand.to_oemol())

            for tautomer in oequacpac.OEGetReasonableTautomers(
                oemol, tautomer_opts, self.pka_norm
            ):
                fmol = oechem.OEMol(tautomer)
                # copy the ligand properties over to the new molecule, we may want to have more fine grained control over this
                # down the track.
                tautomer_ligand = Ligand.from_oemol(
                    fmol, **parent_ligand.dict(exclude={"provenance", "data"})
                )
                # only add the expansion tag to new molecules
                if tautomer_ligand.fixed_inchikey != parent_ligand.fixed_inchikey:
                    tautomer_ligand.set_expansion(
                        parent=parent_ligand, provenance=provenance
                    )
                    expanded_states.append(tautomer_ligand)
                else:
                    expanded_states.append(parent_ligand)

            # return the input ligand
            if parent_ligand not in expanded_states:
                expanded_states.append(parent_ligand)

        return expanded_states
