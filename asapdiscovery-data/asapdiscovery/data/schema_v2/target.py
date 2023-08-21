from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union  # noqa: F401

from asapdiscovery.data.openeye import (
    bytes64_to_oedu,
    load_openeye_design_unit,
    load_openeye_pdb,
    oechem,
    oedu_to_bytes64,
    oemol_to_pdb_string,
    pdb_string_to_oemol,
    save_openeye_design_unit,
)
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.modeling.modeling import split_openeye_mol
from asapdiscovery.modeling.schema import MoleculeFilter
from pydantic import Field, root_validator

from .schema_base import (
    DataModelAbstractBase,
    DataStorageType,
    check_strings_for_equality_with_exclusion,
    schema_dict_get_val_overload,
    write_file_directly,
)


class InvalidTargetError(ValueError):
    ...


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


class Target(DataModelAbstractBase):
    """
    Schema for a Target, wrapper around a PDB file
    """

    target_name: str = Field(None, description="The name of the target")

    ids: Optional[TargetIdentifiers] = Field(
        None,
        description="TargetIdentifiers Schema for identifiers associated with this target",
    )

    data: str = Field(
        "",
        description="PDB file stored as a string to hold internal data state",
        repr=False,
    )
    data_format: DataStorageType = Field(
        DataStorageType.pdb,
        description="Enum describing the data storage method",
        allow_mutation=False,
    )

    @root_validator(pre=True)
    @classmethod
    def _validate_at_least_one_id(cls, v):
        # check if skip validation
        if v.get("_skip_validate_ids"):
            return v
        else:
            ids = v.get("ids")
            compound_name = v.get("target_name")
            # check if all the identifiers are None, sometimes when this is called from
            # already instantiated ligand we need to be able to handle a dict and instantiated class
            if compound_name is None:
                if ids is None or all(
                    [not v for v in schema_dict_get_val_overload(ids)]
                ):
                    raise ValueError(
                        "At least one identifier must be provide, or target_name must be provided"
                    )
        return v

    @classmethod
    def from_pdb(
        cls, pdb_file: Union[str, Path], target_chains=[], ligand_chain="", **kwargs
    ) -> "Target":
        kwargs.pop("data", None)
        # directly read in data
        # First load full complex molecule
        complex_mol = load_openeye_pdb(pdb_file)

        # Split molecule into parts using given chains
        mol_filter = MoleculeFilter(
            protein_chains=target_chains, ligand_chain=ligand_chain
        )
        split_dict = split_openeye_mol(complex_mol, mol_filter)

        return cls.from_oemol(split_dict["prot"], **kwargs)

    def to_pdb(self, filename: Union[str, Path]) -> None:
        # directly write out data
        write_file_directly(filename, self.data)

    @classmethod
    def from_oemol(cls, mol: oechem.OEMol, **kwargs) -> "Target":
        kwargs.pop("data", None)
        pdb_str = oemol_to_pdb_string(mol)
        return cls(data=pdb_str, **kwargs)

    def to_oemol(self) -> oechem.OEMol:
        return pdb_string_to_oemol(self.data)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Target):
            return NotImplemented
        # check if the data is the same
        # but exclude the MASTER record as this is not always in the SAME PLACE
        # for some strange reason
        return check_strings_for_equality_with_exclusion(
            self.data, other.data, "MASTER"
        )

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


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
