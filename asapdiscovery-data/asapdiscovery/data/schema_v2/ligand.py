import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union  # noqa: F401

from pydantic import Field, root_validator, validator

from asapdiscovery.data.openeye import (
    _set_SD_data_repr,
    clear_SD_data,
    get_SD_data,
    load_openeye_sdf,
    oechem,
    oemol_to_inchi,
    oemol_to_inchikey,
    oemol_to_sdf_string,
    oemol_to_smiles,
    sdf_string_to_oemol,
    smiles_to_oemol,
)
from asapdiscovery.data.schema_v2.identifiers import LigandIdentifiers

from .experimental import ExperimentalCompoundData
from .schema_base import (
    DataModelAbstractBase,
    schema_dict_get_val_overload,
    write_file_directly,
)


class InvalidLigandError(ValueError):
    ...


# Ligand Schema
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

    compound_name: Optional[str] = Field(None, description="Name of compound")
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
        ...,
        description="SDF file stored as a string to hold internal data state",
        repr=False,
    )
    data_format: Literal["sdf"] = Field(
        "sdf",
        description="Enum describing the data storage method",
        const=True,
        allow_mutation=False,
    )

    @root_validator(pre=True)
    @classmethod
    def _validate_at_least_one_id(cls, v):
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

    def __hash__(self):
        return self.json().__hash__()

    def __eq__(self, other: "Ligand") -> bool:
        return self.data_equal(other)

    def data_equal(self, other: "Ligand") -> bool:
        # Take out the header block since those aren't really important in checking
        # equality
        return "\n".join(self.data.split("\n")[2:]) == "\n".join(
            other.data.split("\n")[2:]
        )

    @classmethod
    def from_oemol(cls, mol: oechem.OEMol, **kwargs) -> "Ligand":
        """
        Create a Ligand from an OEMol extracting all SD tags into the internal model
        """
        kwargs.pop("data", None)
        sd_tags = get_SD_data(mol)
        for key, value in sd_tags.items():
            try:
                # check to see if we have JSON of a model field
                kwargs[key] = json.loads(value)
            except json.JSONDecodeError:
                kwargs[key] = value

        # extract all info as a tag if it has no field on the model
        tags = {
            (key, value)
            for key, value in kwargs.items()
            if key not in cls.__fields__.keys()
        }
        kwargs["tags"] = tags
        # clean the sdf data for the internal model
        sdf_str = oemol_to_sdf_string(clear_SD_data(mol))
        return cls(data=sdf_str, **kwargs)

    def to_oemol(self) -> oechem.OEGraphMol:
        """
        Convert the current molecule state to an OEMol including all fields as SD tags
        """
        mol = sdf_string_to_oemol(self.data)
        data = {}
        for key in self.__fields__.keys():
            if key not in ["data", "tags"]:
                field = getattr(self, key)
                try:
                    data[key] = field.json()
                except AttributeError:
                    if field is not None:
                        data[key] = str(getattr(self, key))
        # dump tags as separate items
        if self.tags is not None:
            data.update({k: v for k, v in self.tags.items()})
        mol = _set_SD_data_repr(mol, data)
        return mol

    @classmethod
    def from_smiles(cls, smiles: str, **kwargs) -> "Ligand":
        """
        Create a Ligand from a SMILES string
        """
        kwargs.pop("data", None)
        mol = smiles_to_oemol(smiles)
        return cls.from_oemol(mol, **kwargs)

    @property
    def smiles(self) -> str:
        """
        Get the SMILES string for the ligand
        """
        mol = self.to_oemol()
        return oemol_to_smiles(mol)

    @classmethod
    def from_inchi(cls, inchi: str, **kwargs) -> "Ligand":
        """
        Create a Ligand from an InChI string
        """
        kwargs.pop("data", None)
        mol = oechem.OEGraphMol()
        oechem.OEInChIToMol(mol, inchi)
        return cls.from_oemol(mol=mol, **kwargs)

    @property
    def inchi(self) -> str:
        """
        Get the InChI string for the ligand
        """
        mol = self.to_oemol()
        return oemol_to_inchi(mol=mol, fixed_hydrogens=False)

    @property
    def fixed_inchi(self) -> str:
        """
        Returns
        -------
            The fixed hydrogen inchi for the ligand.
        """
        mol = self.to_oemol()
        return oemol_to_inchi(mol=mol, fixed_hydrogens=True)

    @property
    def inchikey(self) -> str:
        """
        Get the InChIKey string for the ligand
        """
        mol = self.to_oemol()
        return oemol_to_inchikey(mol=mol, fixed_hydrogens=False)

    @property
    def fixed_inchikey(self) -> str:
        """
        Returns
        -------
         The fixed hydrogen layer inchi key for the ligand
        """
        mol = self.to_oemol()
        return oemol_to_inchikey(mol=mol, fixed_hydrogens=True)

    @classmethod
    def from_sdf(
        cls,
        sdf_file: Union[str, Path],
        **kwargs,
    ) -> "Ligand":
        """
        Read in a ligand from an SDF file extracting all possible SD data into internal fields.

        Parameters
        ----------
        sdf_file : Union[str, Path]
            Path to the SDF file
        """
        oemol = load_openeye_sdf(sdf_file)
        return cls.from_oemol(oemol, **kwargs)

    def to_sdf(self, filename: Union[str, Path]) -> None:
        """
        Write out the ligand to an SDF file with all attributes stored as SD tags

        Parameters
        ----------
        filename : Union[str, Path]
            Path to the SDF file

        """
        mol = self.to_oemol()
        write_file_directly(filename, oemol_to_sdf_string(mol))

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

    def to_sdf_str(self) -> str:
        """
        Set the SD data for a ligand to a string representation of the data
        that can be written out to an SDF file
        """
        mol = self.to_oemol()
        return oemol_to_sdf_string(mol)

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

    # def _clear_internal_SD_data(self) -> None:
    #     """
    #     Remove SD data from the internal SDF string
    #     """
    #     mol = sdf_string_to_oemol(self.data)
    #     mol = clear_SD_data(mol)
    #     self.data = oemol_to_sdf_string(mol)

    # def flush_attrs_to_SD_data(self) -> str:
    #     """
    #     Flush all attributes to SD data returning the whole new SDF string
    #     """
    #     data = self.dict()
    #     # remove keys that should not be in SD data
    #     data.pop("data")
    #     data.pop("data_format")
    #     if self.ids is not None:
    #         data["ids"] = self.ids.to_SD_tags()
    #     if self.experimental_data is not None:
    #         # Cannot use nested dicts in SD data so we pop the values in experimental_data to a separate key
    #         # `experimental_data_values`
    #         (
    #             data["experimental_data"],
    #             data["experimental_data_values"],
    #         ) = self.experimental_data.to_SD_tags()
    #
    #     # get reserved attribute names
    #     if self.tags is not None:
    #         data.update({k: v for k, v in self.tags.items()})
    #     data.pop("tags")
    #     # update SD data
    #     sdf_str = self._set_SD_data_repr_to_str(data)
    #     return sdf_str

    # def pop_attrs_from_SD_data(self) -> None:
    #     """Pop all attributes from SD data, reserializing the object"""
    #     sd_data = _get_SD_data_to_object(self.to_oemol())
    #     data = self.dict()
    #     # update keys from SD data
    #     data.update(sd_data)
    #
    #     # put experimental data values back into experimental_data if they exist
    #     if "experimental_data_values" in data:
    #         data["experimental_data"]["experimental_data"] = data.pop(
    #             "experimental_data_values"
    #         )
    #     # get reserved attribute names
    #     reser_attr_names = [attr.name for attr in self.__fields__.values()]
    #     # push all non reserved attribute names to tags
    #     data["tags"].update(
    #         {k: v for k, v in data.items() if k not in reser_attr_names}
    #     )
    #     # reinitialise object
    #     self.__init__(**data)

    # def make_parent_tag(
    #     self, provenance: Optional[dict[str, Any]] = None
    # ) -> StateExpansionTag:
    #     """
    #     Create a new expansion tag for the ligand, set it and return it
    #
    #     Returns
    #     -------
    #     StateExpansionTag
    #         The new expansion tag
    #     """
    #     tag = StateExpansionTag.parent(self.inchi, provenance=provenance)
    #     self.expansion_tag = tag
    #     return tag

    # # put experimental data values back into experimental_data if they exist
    # if "experimental_data_values" in data:
    #     data["experimental_data"]["experimental_data"] = data.pop(
    #         "experimental_data_values"
    #     )
    # # get reserved attribute names
    # reser_attr_names = [attr.name for attr in self.__fields__.values()]
    # # push all non reserved attribute names to tags
    # data["tags"].update(
    #     {k: v for k, v in data.items() if k not in reser_attr_names}
    # )


class ReferenceLigand(Ligand):
    target_name: Optional[str] = None
