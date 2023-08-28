import abc
from pathlib import Path
from typing import Literal, Optional

import yaml
from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.utils import seqres_to_res_list
from asapdiscovery.modeling.modeling import (
    make_design_unit,
    mutate_residues,
    spruce_protein,
    superpose_molecule,
)
from pydantic import BaseModel, Field, root_validator


class ProteinPrepperBase(BaseModel):
    """
    Base class for protein preppers.
    """

    prepper_type: Literal["ProteinPrepperBase"] = Field(
        "ProteinPrepperBase", description="The type of prepper to use"
    )

    class Config:
        arbitrary_types_allowed = True

    @abc.abstractmethod
    def _prep(self, *args, **kwargs) -> oechem.OEDesignUnit:
        ...

    def prep(self, *args, **kwargs) -> oechem.OEDesignUnit:
        return self._prep(*args, **kwargs)

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]:
        ...


class ProteinPrepper(ProteinPrepperBase):
    """
    Protein prepper class that uses OESpruce to prepare a protein for docking.
    """

    prepper_type: Literal["ProteinPrepper"] = Field(
        "ProteinPrepper", description="The type of prepper to use"
    )

    align: Optional[oechem.OEMol] = Field(
        None, description="Reference structure to align to."
    )
    ref_chain: Optional[str] = Field(None, description="Chain ID to align to.")
    active_site_chain: Optional[str] = Field(None, description="Chain ID to align to.")
    seqres_yaml: Optional[Path] = Field(
        None, description="Path to seqres yaml to mutate to."
    )
    loop_db: Optional[Path] = Field(
        None, description="Path to loop database to use for prepping"
    )
    oe_active_site_residue: Optional[str] = Field(
        None, description="OE formatted string of active site residue to use"
    )

    @root_validator
    @classmethod
    def _check_align_and_chain_info(cls, values):
        """
        Check that align and chain info is provided correctly.
        """
        align = values.get("align")
        ref_chain = values.get("ref_chain")
        active_site_chain = values.get("active_site_chain")
        if align and not ref_chain:
            raise ValueError("Must provide ref_chain if align is provided")
        if align and not active_site_chain:
            raise ValueError("Must provide active_site_chain if align is provided")
        return values

    def _prep(self, prot: oechem.OEMol) -> oechem.OEDesignUnit:
        """
        Prepares a protein for docking using OESpruce.
        """
        # align
        if self.align:
            prot, _ = superpose_molecule(
                self.align, prot, self.ref_chain, self.active_site_chain
            )

        # mutate residues
        if self.seqres_yaml:
            with open(self.seqres_yaml) as f:
                seqres_dict = yaml.safe_load(f)
            seqres = seqres_dict["SEQRES"]
            res_list = seqres_to_res_list(seqres)
            prot = mutate_residues(prot, res_list, place_h=True)
            protein_sequence = " ".join(res_list)
        else:
            seqres = None
            protein_sequence = None

        # spruce protein
        success, spruce_error_message, spruced = spruce_protein(
            initial_prot=prot,
            protein_sequence=protein_sequence,
            loop_db=str(self.loop_db),
        )

        if not success:
            raise ValueError(f"Prep failed, with error message: {spruce_error_message}")

        success, du = make_design_unit(
            spruced,
            site_residue=self.oe_active_site_residue,
            protein_sequence=protein_sequence,
        )
        if not success:
            raise ValueError("Failed to make design unit.")

        return du

    def provenance(self) -> dict[str, str]:
        return {"prepper_type": self.prepper_type}
