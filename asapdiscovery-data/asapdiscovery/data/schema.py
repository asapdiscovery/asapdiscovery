from datetime import date
from typing import Optional, Union
from pydantic import BaseModel, ValidationError, validator, Field
from .validation import is_valid_smiles

# From FAH ###################################
class Model(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"


class ExperimentalCompoundData(Model):
    compound_id: str = Field(
        None,
        description="The unique compound identifier (PostEra or enumerated ID)",
    )

    smiles: str = Field(
        None,
        description="OpenEye canonical isomeric SMILES string defining suspected SMILES of racemic mixture (with unspecified stereochemistry) or specific enantiopure compound (if racemic=False); may differ from what is registered under compound_id.",
    )

    racemic: bool = Field(
        False,
        description="If True, this experiment was performed on a racemate; if False, the compound was enantiopure.",
    )

    achiral: bool = Field(
        False,
        description="If True, this compound has no chiral centers or bonds, by definition enantiopure",
    )

    absolute_stereochemistry_enantiomerically_pure: bool = Field(
        False,
        description="If True, the compound was enantiopure and stereochemistry recorded in SMILES is correct",
    )

    relative_stereochemistry_enantiomerically_pure: bool = Field(
        False,
        description="If True, the compound was enantiopure, but unknown if stereochemistry recorded in SMILES is correct",
    )

    date_created: date = Field(None, description="Date the molecule was created.")

    experimental_data: dict[str, float] = Field(
        dict(),
        description='Experimental data fields, including "pIC50" and uncertainty (either "pIC50_stderr" or  "pIC50_{lower|upper}"',
    )


class ExperimentalCompoundDataUpdate(Model):
    """A bundle of experimental data for compounds (racemic or enantiopure)."""

    compounds: list[ExperimentalCompoundData]


########################################


class CrystalCompoundData(BaseModel):
    smiles: str = Field(
        None,
        description="OpenEye canonical isomeric SMILES string defining suspected SMILES of racemic mixture (with unspecified stereochemistry) or specific enantiopure compound (if racemic=False); may differ from what is registered under compound_id.",
    )

    compound_id: str = Field(
        None, description="The unique compound identifier of the ligand."
    )

    dataset: str = Field(
        None, description="Dataset name from Fragalysis (name of structure)."
    )

    str_fn: str = Field(None, description="Filename of the PDB structure.")

    sdf_fn: str = Field(None, description="Filename of the SDF file")
    active_site_chain: str = Field(
        None, description="Chain identifying the active site of interest."
    )
    output_name: str = Field(None, description="Name of output structure.")
    active_site: str = Field(None, description="OpenEye formatted active site residue.")
    oligomeric_state: str = Field(
        None, description="Oligomeric state of the asymmetric unit."
    )
    chains: list = Field(None, description="List of chainids in the asymmetric unit.")
    protein_chains: list = Field(
        None, description="List of chains corresponding to protein residues."
    )

    series: str = Field(
        None,
        description="Name of COVID Moonshot series associated with this molecule",
    )


class PDBStructure(Model):
    pdb_id: str = Field(None, description="PDB identification code.")
    str_fn: str = Field(None, description="Filename of local PDB structure.")


class EnantiomerPair(Model):
    active: ExperimentalCompoundData = Field(description="Active enantiomer.")
    inactive: ExperimentalCompoundData = Field(description="Inactive enantiomer.")


class EnantiomerPairList(Model):
    pairs: list[EnantiomerPair]


class ProvenanceBase(Model):
    ...


#########################################




class Ligand(BaseModel):
    smiles: str
    id: str = Field(None, description="the compound identifier")
    vc_id_postera: str = Field(None, description="the PostERA master compound ID")
    moonshot_compound_id: str = Field(None, description="the Moonshot compound ID")
    target_id: str = Field(None, description="the target protein ID")
    source: str = None

    ligand_provenance: Optional[ProvenanceBase] = None

    @validator("smiles")
    def smiles_must_be_valid(cls, v):
        if not is_valid_smiles(v):
            raise ValueError("Invalid SMILES string")

    def to_sdf():
        ...

    def to_smiles():
        ...

    def to_oemol():
        ...

    @staticmethod
    def from_smiles():
        ...

    @staticmethod
    def from_sdf():
        ...

    @staticmethod
    def from_design_unit():
        ...

    @staticmethod
    def from_pdb():
        ...
    
    @staticmethod
    def from_multiligand_sdf():
        ...

