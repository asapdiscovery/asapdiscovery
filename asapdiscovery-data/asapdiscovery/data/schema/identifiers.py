from typing import Optional, Literal, Any

from asapdiscovery.data.schema.schema_base import DataModelAbstractBase
from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags
from pydantic import Field, validator, BaseModel


class LigandIdentifiers(DataModelAbstractBase):
    """
    This is a schema for the identifiers associated with a ligand, optional identifiers are used as part of the ASAP
    workflow. Non-optional identifiers track the state of the molecule to ensure it is always consistent between the
    SDF data and the original input ligand.

    Parameters
    ----------
    moonshot_compound_id : Optional[str], optional
        Moonshot compound ID, by default None
    manifold_api_id : Optional[UUID], optional
        Unique ID from Postera Manifold API, by default None
    manifold_vc_id : Optional[str], optional
        Unique VC ID (virtual compound ID) from Postera Manifold, by default None
    compchem_id : Optional[UUID4], optional
        Unique ID for P5 compchem reference, unused for now, by default None
    """

    moonshot_compound_id: Optional[str] = Field(
        None, description="Moonshot compound ID"
    )
    manifold_api_id: Optional[str] = Field(
        None, description="Unique ID from Postera Manifold API"
    )
    manifold_vc_id: Optional[str] = Field(
        None, description="Unique VC ID (virtual compound ID) from Postera Manifold"
    )
    compchem_id: Optional[str] = Field(
        None, description="Unique ID for P5 compchem reference, unused for now"
    )

    class Config:
        allow_mutation = False

    @validator("manifold_api_id", "compchem_id", pre=True)
    def cast_uuids(cls, v):
        """
        Cast UUIDS to string
        """
        if v is None:
            return None
        else:
            return str(v)


class LigandProvenance(DataModelAbstractBase):
    class Config:
        allow_mutation = False

    isomeric_smiles: str = Field(
        ..., description="The canonical isomeric smiles pattern for the molecule."
    )
    inchi: str = Field(..., description="The standard inchi for the input molecule.")
    inchi_key: str = Field(
        ..., description="The standard inchikey for the input molecule."
    )
    fixed_inchi: str = Field(
        ...,
        description="The non-standard fixed hydrogen layer inchi for the input molecule.",
    )
    fixed_inchikey: str = Field(
        ...,
        description="The non-standard fixed hydrogen layer inchi key for the input molecule.",
    )


class TargetIdentifiers(DataModelAbstractBase):
    """
    Identifiers for a Target
    """

    target_type: Optional[TargetTags] = Field(
        None,
        description="Tag describing the target type e.g SARS-CoV-2-Mpro, etc.",
    )

    fragalysis_id: Optional[str] = Field(
        None, description="The Fragalysis ID of the target if applicable"
    )

    pdb_code: Optional[str] = Field(
        None, description="The PDB code of the target if applicable"
    )


class ChargeProvenance(BaseModel):
    """A simple model to record the provenance of the local charging method."""

    class Config:
        allow_mutation = False

    type: Literal["ChargeProvenance"] = "ChargeProvenance"

    protocol: dict[str, Any] = Field(
        ..., description="The protocol and settings used to generate the local charges."
    )
    provenance: dict[str, str] = Field(
        ..., description="The versions of the software used to generate the local charges."
    )
