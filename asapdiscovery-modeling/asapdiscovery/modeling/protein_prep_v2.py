import abc

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from pathlib import Path
from asapdiscovery.data.openeye import oechem



class ProteinPrepperBase(BaseModel):
    prepper_type: Literal["ProteinPrepperBase"] = Field(
        "ProteinPrepperBase", description="The type of prepper to use"
    )

    @abc.abstractmethod
    def _prep(self, *args, **kwargs):
        ...

    def prep(self, *args, **kwargs):
        return self._prep(*args, **kwargs)

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]:
        ...


class ProteinPrepper(ProteinPrepperBase):
    prepper_type: Literal["ProteinPrepper"] = Field(
        "ProteinPrepper", description="The type of prepper to use")
    
    align: Optional[oechem.OEMol] = Field(None, description="Reference structure to align to.")
    ref_chain: Optional[str] = Field(None, description="Chain ID to align to.")
    active_site_chain: Optional[str] = Field(None, description="Chain ID to align to.")
    seqres_yaml: Optional[Path] = Field(None, description="Path to seqres yaml to mutate to.")
    loop_db: Optional[Path] = Field(None, description="Path to loop database to use for prepping")
    oe_active_site_residue: Optional[str] = Field(None, description="OE formatted string of active site residue to use")


    def _prep(complex_oemol: oechem.OEMol):
        """
        Prepares a protein using the given kwargs

        Parameters
        ----------
        target_oemol : oechem.OEMol
            The protein to prepare

        Returns
        -------
        oechem.OEMol
            The prepared protein
        """
