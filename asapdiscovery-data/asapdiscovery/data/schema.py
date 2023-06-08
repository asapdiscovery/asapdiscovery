import json
import pickle as pkl
from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import Union
from pathlib import Path


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


class Data(BaseModel):
    pass


class CrystalCompoundData(Data):
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


class Dataset(BaseModel):
    class Config:
        extra = "forbid"

    data_type = Data
    iterable: list[data_type]

    def to_csv(self, fn):
        df = pd.DataFrame([vars(data) for data in self.iterable])

        df.to_csv(fn, index=False)

    def to_pkl(self, fn):
        with open(fn, "wb") as file:
            pkl.dump(self, file)

    def to_json(self, fn: Union[str, Path]):
        import pdb as pdb_debug

        pdb_debug.set_trace()
        to_write = self.dict()
        if not isinstance(to_write, dict):
            raise TypeError(
                f"Failed to construct dictionary from {self}, got {type(to_write)} instead."
            )
        with open(fn, "w") as file:
            json.dump(to_write, file)

    @classmethod
    def from_pkl(cls, fn):
        with open(fn, "rb") as file:
            return pkl.load(file)

    @classmethod
    def from_json(cls, fn):
        with open(fn) as file:
            return cls(**json.load(file))

    @classmethod
    def from_csv(cls, fn):
        df = pd.read_csv(fn)
        df = df.replace(np.nan, None)

        return cls(iterable=[cls.data_type(**row) for row in df.to_dict("records")])

    @classmethod
    def from_list(cls, list_of_data_objects):
        return cls(iterable=[data for data in list_of_data_objects])


class CrystalCompoundDataset(Dataset):
    data_type = CrystalCompoundData
    iterable = list[data_type]


class PDBStructure(Model):
    pdb_id: str = Field(None, description="PDB identification code.")
    str_fn: str = Field(None, description="Filename of local PDB structure.")


class EnantiomerPair(Model):
    active: ExperimentalCompoundData = Field(description="Active enantiomer.")
    inactive: ExperimentalCompoundData = Field(description="Inactive enantiomer.")


class EnantiomerPairList(Model):
    pairs: list[EnantiomerPair]
