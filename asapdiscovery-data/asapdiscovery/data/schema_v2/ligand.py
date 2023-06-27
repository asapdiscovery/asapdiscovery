from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union  # noqa: F401
from uuid import UUID

from asapdiscovery.data.openeye import (
    get_SD_data,
    oechem,
    oemol_to_inchi,
    oemol_to_inchikey,
    oemol_to_sdf_string,
    oemol_to_smiles,
    print_SD_Data,
    sdf_string_to_oemol,
    set_SD_data,
    smiles_to_oemol,
)
from asapdiscovery.data.schema import ExperimentalCompoundData
from pydantic import UUID4, Field

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
        None, description="Unique ID for P5 compchem reference"
    )


class Ligand(DataModelAbstractBase):
    """
    Schema for a Ligand
    """

    compound_name: str = Field(None, description="Name of compound")
    ids: Optional[LigandIdentifiers] = Field(
        None,
        description="LigandIdentifiers Schema for identifiers associated with this ligand",
    )
    experimental_data: Optional[ExperimentalCompoundData] = Field(
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
        cls, mol: oechem.OEMol, compound_name: Optional[str] = None, **kwargs
    ) -> "Ligand":
        sdf_str = oemol_to_sdf_string(mol)
        return cls(data=sdf_str, compound_name=compound_name, **kwargs)

    def to_oemol(self) -> oechem.OEMol:
        mol = sdf_string_to_oemol(self.data)
        return mol

    @classmethod
    def from_smiles(
        cls, smiles: str, compound_name: Optional[str] = None, **kwargs
    ) -> "Ligand":
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
        cls, sdf_file: Union[str, Path], compound_name: Optional[str] = None, **kwargs
    ) -> "Ligand":
        # directly read in data
        sdf_str = read_file_directly(sdf_file)
        return cls(data=sdf_str, compound_name=compound_name, **kwargs)

    def to_sdf(self, filename: Union[str, Path]) -> None:
        # directly write out data
        write_file_directly(filename, self.data)

    def set_SD_data(self, data: dict[str, str]) -> None:
        mol = sdf_string_to_oemol(self.data)
        mol = set_SD_data(mol, data)
        self.data = oemol_to_sdf_string(mol)

    def get_SD_data(self) -> dict[str, str]:
        mol = sdf_string_to_oemol(self.data)
        return get_SD_data(mol)

    def print_SD_Data(self) -> None:
        mol = sdf_string_to_oemol(self.data)
        print_SD_Data(mol)


class ReferenceLigand(Ligand):
    target_name: Optional[str] = None
