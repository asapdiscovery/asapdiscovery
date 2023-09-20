from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union  # noqa: F401
from uuid import UUID

from asapdiscovery.data.openeye import (
    _get_SD_data_to_object,
    _set_SD_data_repr,
    clear_SD_data,
    oechem,
    oemol_to_inchi,
    oemol_to_inchikey,
    oemol_to_sdf_string,
    oemol_to_smiles,
    sdf_string_to_oemol,
    smiles_to_oemol,
)
from pydantic import UUID4, Field, root_validator, validator

from .experimental import ExperimentalCompoundData
from .schema_base import (
    DataModelAbstractBase,
    DataStorageType,
    read_file_directly,
    schema_dict_get_val_overload,
    write_file_directly,
)


class InvalidLigandError(ValueError):
    ...


# Ligand Schema


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


class Ligand(DataModelAbstractBase):
    """
    Schema for a Ligand.

    Has first class serialization support for SDF files as well as the typical JSON and dictionary
    serialization.

    Note that equality comparisons are done on the chemical structure data found in the `data` field, not the other fields or the SD Tags in the original SDF
    This means you can change the other fields and still have equality, but changing the chemical structure data will change
    equality.

    You must provide either a compound_name or ids field otherwise the ligand will be invalid.

    Parameters
    ----------
    compound_name : str, optional
        Name of compound, by default None
    ids : Optional[LigandIdentifiers], optional
        LigandIdentifiers Schema for identifiers associated with this ligand, by default None
    experimental_data : Optional[ExperimentalCompoundData], optional
        ExperimentalCompoundData Schema for experimental data associated with the compound, by default None
    tags : dict[str, str], optional
        Dictionary of SD tags, by default {}
    data : str, optional, private
        Chemical structure data from the SDF file stored as a string ""
    data_format : DataStorageType, optional, private, const
        Enum describing the data storage method, by default DataStorageType.sdf
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

    tags: dict[str, str] = Field({}, description="Dictionary of SD tags")

    data: str = Field(
        "",
        description="SDF file stored as a string to hold internal data state",
        repr=False,
    )
    data_format: DataStorageType = Field(
        DataStorageType.sdf,
        description="Enum describing the data storage method",
        const=True,
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
            compound_name = v.get("compound_name")
            # check if all the identifiers are None, sometimes when this is called from
            # already instantiated ligand we need to be able to handle a dict and instantiated class
            if compound_name is None:
                if ids is None or all(
                    [v is None for v in schema_dict_get_val_overload(ids)]
                ):
                    raise ValueError(
                        "At least one identifier must be provide, or compound_name must be provided"
                    )
        return v

    @validator("tags")
    @classmethod
    def _validate_tags(cls, v):
        # check that tags are not reserved attribute names
        reser_attr_names = cls.__fields__.keys()
        for k in v.keys():
            if k in reser_attr_names:
                raise ValueError(f"Tag name {k} is a reserved attribute name")
        return v

    def __eq__(self, other: "Ligand") -> bool:
        return self.data_equal(other)

    def data_equal(self, other: "Ligand") -> bool:
        # Take out the header block since those aren't really important in checking
        # equality
        return "\n".join(self.data.split("\n")[2:]) == "\n".join(
            other.data.split("\n")[2:]
        )

    @classmethod
    def from_oemol(
        cls, mol: oechem.OEMol, compound_name: Optional[str] = None, **kwargs
    ) -> "Ligand":
        """
        Create a Ligand from an OEMol
        """
        sdf_str = oemol_to_sdf_string(mol)
        return cls(data=sdf_str, compound_name=compound_name, **kwargs)

    def to_oemol(self) -> oechem.OEGraphMol:
        """
        Convert to an OEMol
        """
        mol = sdf_string_to_oemol(self.data)
        return mol

    @classmethod
    def from_smiles(
        cls, smiles: str, compound_name: Optional[str] = None, **kwargs
    ) -> "Ligand":
        """
        Create a Ligand from a SMILES string
        """
        mol = smiles_to_oemol(smiles)
        sdf_str = oemol_to_sdf_string(mol)
        return cls(data=sdf_str, compound_name=compound_name, **kwargs)

    @property
    def smiles(self) -> str:
        """
        Get the SMILES string for the ligand
        """
        mol = sdf_string_to_oemol(self.data)
        return oemol_to_smiles(mol)

    @property
    def inchi(self) -> str:
        """
        Get the InChI string for the ligand
        """
        mol = sdf_string_to_oemol(self.data)
        return oemol_to_inchi(mol)

    @property
    def inchikey(self) -> str:
        """
        Get the InChIKey string for the ligand
        """
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
        # we have to skip validation here, because we don't have a bunch of fields as they
        # still need to be read in from the SD tags
        lig = cls(
            data=sdf_str, compound_name=compound_name, _skip_validate_ids=True, **kwargs
        )
        if read_SD_attrs:
            lig.pop_attrs_from_SD_data()
        lig._clear_internal_SD_data()
        # okay now we can validate !
        lig.validate(lig.dict())
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
            Whether to write out attributes as SD tags, by default True

        """
        if write_SD_attrs:
            data_to_write = self.flush_attrs_to_SD_data()
        else:
            data_to_write = self.data
        # directly write out data
        write_file_directly(filename, data_to_write)

    def set_SD_data(self, data: dict[str, str]) -> None:
        """
        Set the SD data for the ligand, uses an update to overwrite existing data in line with
        OpenEye behaviour
        """
        # make sure we don't overwrite any attributes
        for k in data.keys():
            if k in self.__fields__.keys():
                raise ValueError(f"Tag name {k} is a reserved attribute name")
        self.tags.update(data)

    def _set_SD_data_repr_to_str(self, data: dict[str, str]) -> str:
        """
        Set the SD data for a ligand to a string representation of the data
        that can be written out to an SDF file
        """
        mol = sdf_string_to_oemol(self.data)
        mol = _set_SD_data_repr(mol, data)
        sdf_str = oemol_to_sdf_string(mol)
        return sdf_str

    def get_SD_data(self) -> dict[str, str]:
        """
        Get the SD data for the ligand
        """
        return self.tags

    def print_SD_data(self) -> None:
        """
        Print the SD data for the ligand
        """
        print(self.tags)

    def clear_SD_data(self) -> None:
        """
        Clear the SD data for the ligand
        """
        self.tags = {}

    def _clear_internal_SD_data(self) -> None:
        """
        Remove SD data from the internal SDF string
        """
        mol = sdf_string_to_oemol(self.data)
        mol = clear_SD_data(mol)
        self.data = oemol_to_sdf_string(mol)

    def flush_attrs_to_SD_data(self) -> str:
        """
        Flush all attributes to SD data returning the whole new SDF string
        """
        data = self.dict()
        # remove keys that should not be in SD data
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

        # get reserved attribute names
        if self.tags is not None:
            data.update({k: v for k, v in self.tags.items()})
        data.pop("tags")
        # update SD data
        sdf_str = self._set_SD_data_repr_to_str(data)
        return sdf_str

    def pop_attrs_from_SD_data(self) -> None:
        """Pop all attributes from SD data, reserializing the object"""
        sd_data = _get_SD_data_to_object(self.to_oemol())
        data = self.dict()
        # update keys from SD data
        data.update(sd_data)

        # put experimental data values back into experimental_data if they exist
        if "experimental_data_values" in data:
            data["experimental_data"]["experimental_data"] = data.pop(
                "experimental_data_values"
            )
        # get reserved attribute names
        reser_attr_names = [attr.name for attr in self.__fields__.values()]
        # push all non reserved attribute names to tags
        data["tags"].update(
            {k: v for k, v in data.items() if k not in reser_attr_names}
        )
        # reinitialise object
        self.__init__(**data)


class ReferenceLigand(Ligand):
    target_name: Optional[str] = None
