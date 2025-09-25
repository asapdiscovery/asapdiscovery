from asapdiscovery.data.schema.experimental import ExperimentalCompoundData
from pydantic.v1 import BaseModel, Field


class CrystalCompoundData(BaseModel):
    class Config:
        extra = "forbid"

    compound_id: str = Field(
        None, description="The unique compound identifier of the ligand."
    )

    dataset: str = Field(
        None, description="Dataset name from Fragalysis (name of structure)."
    )
    smiles: str = Field(
        None,
        description="OpenEye canonical isomeric SMILES string defining suspected SMILES of racemic mixture (with unspecified stereochemistry) or specific enantiopure compound (if racemic=False); may differ from what is registered under compound_id.",
    )
    str_fn: str = Field(None, description="Filename of the PDB structure.")

    sdf_fn: str = Field(None, description="Filename of the SDF file")


class EnantiomerPair(BaseModel):
    active: ExperimentalCompoundData = Field(description="Active enantiomer.")
    inactive: ExperimentalCompoundData = Field(description="Inactive enantiomer.")


class EnantiomerPairList(BaseModel):
    pairs: list[EnantiomerPair]
