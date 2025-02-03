from typing import Literal

from asapdiscovery.data.backend.openeye import clear_SD_data, oechem, oeomega
from asapdiscovery.data.operators.state_expanders.state_expander import (
    StateExpanderBase,
)
from asapdiscovery.data.schema.ligand import Ligand
from pydantic.v1 import Field


class StereoExpander(StateExpanderBase):
    """
    Expand a molecule to stereoisomers using OpenEye

    Note:
        The input molecule is only included if it is fully defined.
        Input molecules with no possible expansions are passed through without an expansion tag set.
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

    def _expand(self, ligands: list[Ligand]) -> list[Ligand]:
        """
        Expand the stereoisomers of the input molecules.

        Note:
            Input molecules with no possible expansions are passed through without an expansion tag set.

        Parameters
        ----------
        ligands: The list of ligands whose states should be expanded.

        Returns
        -------
            A list of expanded ligand states.

        """
        provenance = self.provenance()
        omegaOpts = oeomega.OEOmegaOptions()
        omega = oeomega.OEOmega(omegaOpts)
        maxcenters = 20
        force_flip = self.stereo_expand_defined
        enum_nitrogen = (
            False  # WARNING: This creates multiple microstates with same SMILES if True
        )
        warts = False  # add suffix for stereoisomers

        enantiomers = []
        for parent_ligand in ligands:
            # need to clear the SD data otherwise the provenance will break
            oemol = clear_SD_data(parent_ligand.to_oemol())
            for enantiomer in oeomega.OEFlipper(
                oemol, maxcenters, force_flip, enum_nitrogen, warts
            ):
                enantiomer = oechem.OEMol(enantiomer)
                omega.Build(
                    enantiomer
                )  # a single conformer needs to be built to fully define stereochemistry
                enantiomer_ligand = Ligand.from_oemol(
                    enantiomer, **parent_ligand.dict(exclude={"provenance", "data"})
                )
                # if the ligand is the parent ie no possible expansions don't tag it
                if enantiomer_ligand.fixed_inchikey == parent_ligand.fixed_inchikey:
                    enantiomers.append(parent_ligand)
                else:
                    enantiomer_ligand.set_expansion(
                        parent_ligand, provenance=provenance
                    )
                    enantiomers.append(enantiomer_ligand)

        return enantiomers
