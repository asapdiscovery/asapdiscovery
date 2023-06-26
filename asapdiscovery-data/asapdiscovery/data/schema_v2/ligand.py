from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union  # noqa: F401

from asapdiscovery.data.openeye import (
    get_SD_data,
    get_SD_data_dict,
    oechem,
    oemol_to_inchi,
    oemol_to_inchikey,
    oemol_to_sdf_string,
    oemol_to_smiles,
    print_SD_Data,
    sdf_string_to_oemol,
    set_SD_data,
    set_SD_data_dict,
    smiles_to_oemol,
)
from asapdiscovery.data.schema import ExperimentalCompoundData
from pydantic import Field

from .schema_base import (
    DataModelAbstractBase,
    DataStorageType,
    read_file_directly,
    write_file_directly,
)


class InvalidLigandError(ValueError):
    ...


# Ligand Schema


class LigandIdentifiers(DataModelAbstractBase):
    """
    Identifiers for a Ligand
    """

    moonshot_compound_id: str = Field(None, description="Moonshot compound ID")
    postera_vc_id: str | None = Field(None, description="Unique VC ID from Postera")


class Ligand(DataModelAbstractBase):
    """
    Schema for a Ligand
    """

    compound_name: str = Field(None, description="Name of compound")
    ids: LigandIdentifiers | None = Field(
        None,
        description="LigandIdentifiers Schema for identifiers associated with this ligand",
    )
    experimental_data: ExperimentalCompoundData | None = Field(
        None,
        description="ExperimentalCompoundData Schema for experimental data associated with the compound",
    )
    data: str = Field(
        "",
        description="SDF file stored as a string to hold internal data state",
        repr=False,
    )
    data_format: DataStorageType = Field(
        DataStorageType.sdf,
        description="Enum describing the data storage method",
        allow_mutation=False,
    )

    @classmethod
    def from_oemol(
        cls, mol: oechem.OEMol, compound_name: str | None = None, **kwargs
    ) -> Ligand:
        sdf_str = oemol_to_sdf_string(mol)
        return cls(data=sdf_str, compound_name=compound_name, **kwargs)

    def to_oemol(self) -> oechem.OEMol:
        mol = sdf_string_to_oemol(self.data)
        return mol

    @classmethod
    def from_smiles(
        cls, smiles: str, compound_name: str | None = None, **kwargs
    ) -> Ligand:
        mol = smiles_to_oemol(smiles)
        sdf_str = oemol_to_sdf_string(mol)
        return cls(data=sdf_str, compound_name=compound_name, **kwargs)

    @property
    def smiles(self) -> str:
        mol = sdf_string_to_oemol(self.data)
        return oemol_to_smiles(mol)

    @property
    def inchi(self) -> str:
        mol = sdf_string_to_oemol(self.data)
        return oemol_to_inchi(mol)

    @property
    def inchikey(self) -> str:
        mol = sdf_string_to_oemol(self.data)
        return oemol_to_inchikey(mol)

    @classmethod
    def from_sdf(
        cls, sdf_file: str | Path, compound_name: str | None = None, **kwargs
    ) -> Ligand:
        # directly read in data
        sdf_str = read_file_directly(sdf_file)
        return cls(data=sdf_str, compound_name=compound_name, **kwargs)

    def to_sdf(self, filename: str | Path) -> None:
        # directly write out data
        write_file_directly(filename, self.data)

    def set_SD_data(self, key: str, value: str) -> None:
        mol = sdf_string_to_oemol(self.data)
        mol = set_SD_data(mol, key, value)
        self.data = oemol_to_sdf_string(mol)

    def set_SD_data_dict(self, data: dict[str, str]) -> None:
        mol = sdf_string_to_oemol(self.data)
        mol = set_SD_data_dict(mol, data)
        self.data = oemol_to_sdf_string(mol)

    def get_SD_data(self, key: str) -> str:
        mol = sdf_string_to_oemol(self.data)
        return get_SD_data(mol, key)

    def get_SD_data_dict(self) -> dict[str, str]:
        mol = sdf_string_to_oemol(self.data)
        return get_SD_data_dict(mol)

    def print_SD_Data(self) -> None:
        mol = sdf_string_to_oemol(self.data)
        print_SD_Data(mol)


class ReferenceLigand(Ligand):
    target_name: str | None = None
