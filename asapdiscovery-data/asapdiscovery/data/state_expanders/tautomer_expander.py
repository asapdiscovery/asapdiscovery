from typing import Literal

from asapdiscovery.data.openeye import oechem, oequacpac
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.state_expanders.state_expander import StateExpanderBase
from pydantic import Field


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

    def provenance(self) -> dict[str, str]:
        return {
            "expander": self.dict(),
            "oechem": oechem.OEChemGetVersion(),
            "quacpac": oequacpac.OEQuacPacGetVersion(),
        }

    def _expand(self, ligands: list[Ligand]) -> list[Ligand]:
        tautomer_opts = oequacpac.OETautomerOptions()
        tautomer_opts.SetSaveStereo(self.tautomer_save_stereo)
        tautomer_opts.SetCarbonHybridization(self.tautomer_carbon_hybridization)

        expanded_states = []

        for parent_ligand in ligands:
            oemol = parent_ligand.to_oemol()
            parent_ligand.make_parent_tag(provenance=self.provenance())
            for tautomer in oequacpac.OEGetReasonableTautomers(
                oemol, tautomer_opts, self.pka_norm
            ):
                fmol = oechem.OEMol(tautomer)
                # copy the ligand properties over to the new molecule, we may want to have more fine grained control over this
                # down the track.
                tautomer_ligand = Ligand.from_oemol(fmol, **parent_ligand.dict())
                tautomer_ligand.set_parent(parent_ligand, provenance=self.provenance())
                expanded_states.append(tautomer_ligand)

        return expanded_states
