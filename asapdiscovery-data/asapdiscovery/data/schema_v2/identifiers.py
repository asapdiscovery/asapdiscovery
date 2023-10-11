from typing import Any, Dict, Optional, Tuple, Union  # noqa: F401
from uuid import UUID

from asapdiscovery.data.schema_v2.schema_base import DataModelAbstractBase
from pydantic import UUID4, Field


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

    def to_SD_tags(self) -> dict[str, str]:
        """
        Convert to a dictionary of SD tags
        """
        data = self.dict()
        return {str(k): str(v) for k, v in data.items() if v is not None}
