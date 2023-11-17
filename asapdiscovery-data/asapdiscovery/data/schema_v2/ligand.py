import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union  # noqa: F401
from enum import Flag, auto
from asapdiscovery.data.openeye import (
    _set_SD_data_repr,
    clear_SD_data,
    get_SD_data,
    load_openeye_sdf,
    oechem,
    oequacpac,
    oemol_to_inchi,
    oemol_to_inchikey,
    oemol_to_sdf_string,
    oemol_to_smiles,
    sdf_string_to_oemol,
    smiles_to_oemol,
)
from asapdiscovery.data.schema_v2.identifiers import LigandIdentifiers
from asapdiscovery.data.schema_v2.schema_base import DataStorageType
from asapdiscovery.data.state_expanders.expansion_tag import StateExpansionTag
from pydantic import Field, root_validator, validator

from .experimental import ExperimentalCompoundData
from .schema_base import (
    DataModelAbstractBase,
    schema_dict_get_val_overload,
    write_file_directly,
)


class InvalidLigandError(ValueError):
    ...


class ChemicalRelationship(Flag):
    """
    Enum describing the chemical relationship between two ligands.
    Currently not distinguishing between conjugate acids / bases and tautomers, which means that ligands which are
    technically constitutional isomers (i.e. +/- a proton) will be considered tautomers
    """

    DISTINCT = auto()
    IDENTICAL = auto()
    STEREOISOMER = auto()
    TAUTOMER = auto()


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

    expansion_tag: Optional[StateExpansionTag] = Field(
        None,
        description="Expansion tag linking this ligand to its parent in a state expansion if needed",
    )

    tags: dict[str, str] = Field({}, description="Dictionary of SD tags")

    data: str = Field(
        ...,
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
            if key not in ["data", "tags", "data_format"]:
                field = getattr(self, key)
                try:
                    data[key] = field.json()
                except AttributeError:
                    if field is not None:
                        data[key] = str(getattr(self, key))
        # dump the enum using value to get the str repr
        data["data_format"] = self.data_format.value
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
        Get the canonical isomeric SMILES string for the ligand
        """
        mol = self.to_oemol()
        return oemol_to_smiles(mol)

    @property
    def non_iso_smiles(self) -> str:
        """
        Get the non-isomeric canonical SMILES string for the ligand
        """
        mol = self.to_oemol()
        return oemol_to_smiles(mol, isomeric=False)

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

    def to_sdf(self, filename: Union[str, Path], allow_append=False) -> None:
        """
        Write out the ligand to an SDF file with all attributes stored as SD tags

        Parameters
        ----------
        filename : Union[str, Path]
            Path to the SDF file
        allow_append : bool, optional
            Allow appending to the file, by default False

        """
        if allow_append:
            fmode = "a"
        else:
            fmode = "w"
        mol = self.to_oemol()
        write_file_directly(filename, oemol_to_sdf_string(mol), mode=fmode)

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

    def set_expansion(
        self,
        parent: "Ligand",
        provenance: dict[str, Any],
    ) -> None:
        """
        Set the expansion of the ligand with a reference to the parent ligand and the settings used to create the
        expansion.

        Parameters
        ----------
            parent: The parent ligand from which this child was created.
            provenance: The provenance dictionary of the state expander used to create this ligand created via
            `expander.provenance()` where the keys are fields of the expander and the values capture the
            associated settings.
        """
        self.expansion_tag = StateExpansionTag.from_parent(
            parent=parent, provenance=provenance
        )

    @property
    def canonical_tautomer(self) -> "Ligand":
        """
        Get the canonical tautomer of the ligand.
        Not necessarily the most physiologically relevant tautomer, but helpful for comparing ligands.
        """
        mol = self.to_oemol()
        canonical_tautomer = oechem.OEGraphMol()
        if oequacpac.OEGetUniqueProtomer(canonical_tautomer, mol) == True:
            return Ligand.from_oemol(
                compound_name=self.compound_name,
                mol=canonical_tautomer,
                expansion_tag=StateExpansionTag.from_parent(
                    parent=self,
                    provenance={
                        "expander": "oequacpac.OEGetUniqueProtomer",
                        "oechem": oechem.OEChemGetVersion(),
                        "quacpac": oequacpac.OEQuacPacGetVersion(),
                    },
                ),
            )
        else:
            raise ValueError("Unable to generate canonical tautomer")

    @property
    def has_defined_stereo(self) -> bool:
        """
        Check if the ligand has defined stereochemistry
        """
        mol = self.to_oemol()
        return mol.HasPerceived(oechem.OEPerceived_AtomStereo) or mol.HasPerceived(
            oechem.OEPerceived_BondStereo
        )

    def is_chemically_equal(self, other: "Ligand") -> bool:
        """
        Check if the ligand is chemically equal to another ligand using the inchikey.
        Both ligands must both have defined stereochemistry or both not have defined stereochemistry.
        """
        return (
            self.fixed_inchikey == other.fixed_inchikey
            and self.has_defined_stereo == other.has_defined_stereo
        )

    def is_stereoisomer(self, other: "Ligand"):
        """
        Check if the ligand is a possible stereoisomer of another ligand.
        Returns False if the ligands are the same.
        """
        # First check if molecules are the same
        if self.is_chemically_equal(other):
            return False

        return self.non_iso_smiles == other.non_iso_smiles

    def is_tautomer(self, other: "Ligand"):
        """
        Check if the ligand is a tautomer of another ligand, collapsing protonation states.
        Returns False if the ligands are the same or stereoisomers.
        """
        # First check if molecules are the same or just a stereoisomer
        if self.is_chemically_equal(other) or self.is_stereoisomer(other):
            return False
        return self.canonical_tautomer.is_chemically_equal(other.canonical_tautomer)

    def is_related(self, other: "Ligand"):
        """
        Check if the ligand is a steroisomer of a tautomer of another ligand.
        Returns False if the ligands are the same, stereoisomers, or tautomers with the same stereochem.
        """
        # First check if molecules are the same or just a stereoisomer or a tautomer
        if (
            self.is_chemically_equal(other)
            or self.is_stereoisomer(other)
            or self.is_tautomer(other)
        ):
            return False
        else:
            # Get the canonical tautomers and check if the non-iso smiles are the same
            return self.canonical_tautomer.is_stereoisomer(other.canonical_tautomer)

    def get_chemical_relationship(self, other: "Ligand") -> ChemicalRelationship:
        """
        Get the chemical relationship between two ligands
        """
        if self.is_chemically_equal(other):
            return ChemicalRelationship.IDENTICAL
        elif self.is_stereoisomer(other):
            return ChemicalRelationship.STEREOISOMER
        elif self.is_tautomer(other):
            return ChemicalRelationship.TAUTOMER
        elif self.is_related(other):
            return ChemicalRelationship.TAUTOMER | ChemicalRelationship.STEREOISOMER
        else:
            return ChemicalRelationship.DISTINCT


class ReferenceLigand(Ligand):
    target_name: Optional[str] = None


def compound_names_unique(ligands: list[Ligand]) -> bool:
    """
    Check that all the compound names in a list of ligands are unique
    """
    compound_names = [ligand.compound_name for ligand in ligands]
    return len(set(compound_names)) == len(compound_names)


def write_ligands_to_multi_sdf(
    sdf_name: Union[str, Path], ligands: list[Ligand], overwrite=False
):
    """
    Dumb way to do this, but just write out each ligand to the same.
    Alternate way would be to flush each to OEMol and then write out
    using OE but seems convoluted.

    Note that this will overwrite the file if it exists unless overwrite is set to False

    Parameters
    ----------
    sdf_name : Union[str, Path]
        Path to the SDF file
    ligands : list[Ligand]
        List of ligands to write out
    overwrite : bool, optional
        Overwrite the file if it exists, by default False

    Raises
    ------
    FileExistsError
        If the file exists and overwrite is False
    ValueError
        If the sdf_name does not end in .sdf
    """
    sdf_file = Path(sdf_name)
    if sdf_file.exists() and not overwrite:
        raise FileExistsError(f"{sdf_file} exists and overwrite is False")

    elif sdf_file.exists() and overwrite:
        sdf_file.unlink()

    if not sdf_file.suffix == ".sdf":
        raise ValueError("SDF name must end in .sdf")

    for ligand in ligands:
        ligand.to_sdf(sdf_file, allow_append=True)
