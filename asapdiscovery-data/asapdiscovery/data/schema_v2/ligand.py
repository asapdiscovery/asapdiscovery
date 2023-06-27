from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union  # noqa: F401
from uuid import UUID

from asapdiscovery.data.openeye import (
    _get_SD_data_to_object,
    _set_SD_data_repr,
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
from pydantic import UUID4, Field

from .experimental import ExperimentalCompoundData
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
        None, description="Unique ID for P5 compchem reference, unused for now"
    )

    def to_SD_tags(self) -> dict[str, str]:
        """
        Convert to a dictionary of SD tags
        """
        data = self.dict()
        return {str(k): str(v) for k, v in data.items() if v is not None}


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
        cls,
        sdf_file: Union[str, Path],
        compound_name: Optional[str] = None,
        read_SD_attrs: bool = True,
        **kwargs,
    ) -> "Ligand":
        """
        Read in a ligand from an SDF file.
        If read_SD_attrs is True, then SD tags will be read in as attributes, overriding kwargs where double defined.
        If read_SD_attrs is False, then SD tags will not be read in as attributes, and kwargs will be used instead

        Parameters
        ----------
        sdf_file : Union[str, Path]
            Path to the SDF file
        compound_name : Optional[str], optional
            Name of the compound, by default None
        read_SD_attrs : bool, optional
            Whether to read in SD tags as attributes, by default True, overrides kwargs
        """
        # directly read in data
        sdf_str = read_file_directly(sdf_file)
        lig = cls(data=sdf_str, compound_name=compound_name, **kwargs)
        if read_SD_attrs:
            lig.pop_attrs_from_SD_data()
        return lig

    def to_sdf(self, filename: Union[str, Path], write_SD_attrs: bool = True) -> None:
        """
        Write out the ligand to an SDF file
        If write_SD_attrs is True, then SD tags will be written out as attributes
        If write_SD_attrs is False, then SD tags will not be written out as attributes

        Parameters
        ----------
        filename : Union[str, Path]
            Path to the SDF file
        write_SD_attrs : bool, optional
            Whether to write out SD tags as attributes, by default True

        """
        if write_SD_attrs:
            self.flush_attrs_to_SD_data()
        # directly write out data
        write_file_directly(filename, self.data)

    def set_SD_data(self, data: dict[str, str]) -> None:
        mol = sdf_string_to_oemol(self.data)
        mol = set_SD_data(mol, data)
        self.data = oemol_to_sdf_string(mol)

    def _set_SD_data_repr(self, data: dict[str, str]) -> None:
        mol = sdf_string_to_oemol(self.data)
        mol = _set_SD_data_repr(mol, data)
        self.data = oemol_to_sdf_string(mol)

    def get_SD_data(self) -> dict[str, str]:
        mol = sdf_string_to_oemol(self.data)
        return get_SD_data(mol)

    def print_SD_data(self) -> None:
        mol = sdf_string_to_oemol(self.data)
        print_SD_Data(mol)

    def flush_attrs_to_SD_data(self) -> None:
        """Flush all attributes to SD data"""
        data = self.dict()
        # remove keys that are not SD data
        data.pop("data")
        data.pop("data_format")
        if self.ids is not None:
            data["ids"] = self.ids.to_SD_tags()
        if self.experimental_data is not None:
            # Cannot use nested dicts in SD data so we pop the values in experimental_data to a separate key
            # `experimental_data_values`
            (
                data["experimental_data"],
                data["experimental_data_values"],
            ) = self.experimental_data.to_SD_tags()
        # update SD data
        self._set_SD_data_repr(data)

    def pop_attrs_from_SD_data(self) -> None:
        """Pop all attributes from SD data"""
        sd_data = _get_SD_data_to_object(self.to_oemol())
        data = self.dict()
        # update keys from SD data
        data.update(sd_data)

        # put experimental data values back into experimental_data if they exist
        if "experimental_data_values" in data:
            data["experimental_data"]["experimental_data"] = data.pop(
                "experimental_data_values"
            )
        # reconstruct object
        self.__init__(**data)


class ReferenceLigand(Ligand):
    target_name: Optional[str] = None
