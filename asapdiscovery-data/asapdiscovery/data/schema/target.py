import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union  # noqa: F401

from pydantic import Field, root_validator

from asapdiscovery.data.backend.openeye import (
    load_openeye_pdb,
    oechem,
    oemol_to_pdb_string,
    pdb_string_to_oemol,
    split_openeye_mol,
)
from asapdiscovery.data.schema.identifiers import TargetIdentifiers

from .schema_base import (
    DataModelAbstractBase,
    DataStorageType,
    MoleculeFilter,
    check_strings_for_equality_with_exclusion,
    schema_dict_get_val_overload,
    write_file_directly,
)

logger = logging.getLogger(__name__)


class InvalidTargetError(ValueError): ...  # noqa: E701


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

    @property
    def hash(self):
        """Create a hash based on the pdb file contents"""
        import hashlib

        return hashlib.sha256(self.data.encode()).hexdigest()

    @property
    def crystal_symmetry(self):
        """
        Get the crystal symmetry of the target
        """
        return oechem.OEGetCrystalSymmetry(self.to_oemol())


# Re-export PreppedTarget for backward compatibility
from asapdiscovery.modeling.schema import PreppedTarget  # noqa: E402, F401
