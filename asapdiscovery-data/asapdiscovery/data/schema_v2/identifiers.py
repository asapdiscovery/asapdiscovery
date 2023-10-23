from typing import Optional
from uuid import UUID

from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from pydantic import UUID4, Field

from .schema_base import DataModelAbstractBase


class LigandIdentifiers(DataModelAbstractBase):
    """
    This is a schema for the identifiers associated with a ligand

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
    manifold_api_id: Optional[UUID] = Field(
        None, description="Unique ID from Postera Manifold API"
    )
    manifold_vc_id: Optional[str] = Field(
        None, description="Unique VC ID (virtual compound ID) from Postera Manifold"
    )
    compchem_id: Optional[UUID4] = Field(
        None, description="Unique ID for P5 compchem reference, unused for now"
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
