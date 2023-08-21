from typing import Literal

from asapdiscovery.data.openeye import oechem, oequacpac
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.state_expanders.state_expander import StateExpanderBase
from pydantic import Field


class ProtomerExpander(StateExpanderBase):
    """
    Expand a molecule to protomers
    """

    expander_type: Literal["ProtomerExpander"] = "ProtomerExpander"

    def provenance(self) -> dict[str, str]:
        return {
            "expander": self.dict(),
            "oechem": oechem.OEChemGetVersion(),
            "quacpac": oequacpac.OEQuacPacGetVersion(),
        }

    def _expand(self, ligands: list[Ligand]) -> list[Ligand]:
        expanded_states = []

        for parent_ligand in ligands:
            oemol = parent_ligand.to_oemol()
            parent_ligand.make_parent_tag(provenance=self.provenance())
            for protomer in oequacpac.OEGetReasonableProtomers(oemol):
                fmol = oechem.OEMol(protomer)
                # copy the ligand properties over to the new molecule, we may want to have more fine grained control over this
                # down the track.
                protomer_ligand = Ligand.from_oemol(fmol, **parent_ligand.dict())
                protomer_ligand.set_parent(parent_ligand, provenance=self.provenance())
                expanded_states.append(protomer_ligand)

        return expanded_states
