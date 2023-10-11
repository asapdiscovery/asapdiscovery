from typing import Literal

from pydantic import Field

from asapdiscovery.data.openeye import oechem, oequacpac
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.state_expanders.state_expander import StateExpanderBase


class TautomerExpander(StateExpanderBase):
    """
    Expand a molecule to protomers
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
            oemol = parent_ligand.to_oemol()

            for tautomer in oequacpac.OEGetReasonableTautomers(
                oemol, tautomer_opts, self.pka_norm
            ):
                fmol = oechem.OEMol(tautomer)
                # copy the ligand properties over to the new molecule, we may want to have more fine grained control over this
                # down the track.
                tautomer_ligand = Ligand.from_oemol(fmol, **parent_ligand.dict())
                tautomer_ligand.set_expansion(
                    parent=parent_ligand, provenance=provenance
                )
                expanded_states.append(tautomer_ligand)

        return expanded_states
