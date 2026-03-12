import logging
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import Field, root_validator

from asapdiscovery.data.backend.openeye import (
    bytes64_to_oedu,
    load_openeye_design_unit,
    oechem,
    oedu_to_bytes64,
    openeye_perceive_residues,
    save_openeye_design_unit,
    save_openeye_pdb,
    split_openeye_design_unit,
)
from asapdiscovery.data.schema.identifiers import TargetIdentifiers
from asapdiscovery.data.schema.schema_base import (
    DataModelAbstractBase,
    DataStorageType,
    MoleculeComponent,
    MoleculeFilter,
    schema_dict_get_val_overload,
)

# Re-export for backward compatibility
__all__ = ["MoleculeComponent", "MoleculeFilter", "PreppedTarget"]


logger = logging.getLogger(__name__)


class PreppedTarget(DataModelAbstractBase):
    """
    Schema for a PreppedTarget, wrapper around an OpenEye Design Unit
    """

    target_name: str = Field(None, description="The name of the target")

    ids: Optional[TargetIdentifiers] = Field(
        None,
        description="TargetIdentifiers Schema for identifiers associated with this target",
    )

    data: bytes = Field(
        "",
        description="OpenEye oedu file stored as a bytes object **encoded in base64** to hold internal data state",
        repr=False,
    )
    data_format: DataStorageType = Field(
        DataStorageType.b64oedu,
        description="Enum describing the data storage method",
        allow_mutation=False,
    )
    target_hash: str = Field(
        ...,
        description="A unique reproducible hash based on the contents of the pdb file which created the target.",
        allow_mutation=False,
    )

    crystal_symmetry: Optional[Any] = Field(
        None,
        description="bounding box of the target, lost in oedu conversion so can be saved as attribute.",
    )

    @root_validator(pre=True)
    @classmethod
    def _validate_at_least_one_id(cls, v):
        # simpler as we never need to pop attrs off the serialised representation.
        ids = v.get("ids")
        compound_name = v.get("target_name")
        # check if all the identifiers are None
        if compound_name is None:
            if ids is None or all([not v for v in schema_dict_get_val_overload(ids)]):
                raise ValueError(
                    "At least one identifier must be provide, or target_name must be provided"
                )
        return v

    @classmethod
    def from_oedu(cls, oedu: oechem.OEDesignUnit, **kwargs) -> "PreppedTarget":
        kwargs.pop("data", None)
        oedu_bytes = oedu_to_bytes64(oedu)
        return cls(data=oedu_bytes, **kwargs)

    def to_oedu(self) -> oechem.OEDesignUnit:
        return bytes64_to_oedu(self.data)

    @classmethod
    def from_oedu_file(cls, oedu_file: Union[str, Path], **kwargs) -> "PreppedTarget":
        kwargs.pop("data", None)
        oedu = load_openeye_design_unit(oedu_file)
        return cls.from_oedu(oedu=oedu, **kwargs)

    def to_oedu_file(self, filename: Union[str, Path]) -> None:
        oedu = self.to_oedu()
        save_openeye_design_unit(oedu, filename)

    def to_pdb_file(self, filename: str):
        """
        Write the prepared target receptor to PDB file using openeye.
        Parameters
        ----------
        filename: The name of the pdb file the target should be writen to.
        """
        oedu = self.to_oedu()
        _, oe_receptor, _ = split_openeye_design_unit(du=oedu)
        # As advised by Alex <https://github.com/choderalab/asapdiscovery/pull/608#discussion_r1388067468>
        openeye_perceive_residues(oe_receptor)
        save_openeye_pdb(oe_receptor, pdb_fn=filename)

    @property
    def hash(self):
        """Create a hash based on the pdb file contents"""
        import hashlib

        return hashlib.sha256(self.data).hexdigest()
