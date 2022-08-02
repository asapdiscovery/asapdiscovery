from typing import Dict, List

import pandas
from pydantic import BaseModel, Field
import pickle as pkl
import numpy as np

## From FAH #####################################################################
class Model(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"

class ExperimentalCompoundData(Model):

    compound_id: str = Field(
        None, description="The unique compound identifier (PostEra or enumerated ID)"
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

    experimental_data: Dict[str, float] = Field(
        dict(),
        description='Experimental data fields, including "pIC50" and uncertainty (either "pIC50_stderr" or  "pIC50_{lower|upper}"',
    )


class ExperimentalCompoundDataUpdate(Model):
    """A bundle of experimental data for compounds (racemic or enantiopure)."""

    compounds: List[ExperimentalCompoundData]
################################################################################

class CrystalCompoundData(Model):

    smiles: str = Field(
        None,
        description="OpenEye canonical isomeric SMILES string defining suspected SMILES of racemic mixture (with unspecified stereochemistry) or specific enantiopure compound (if racemic=False); may differ from what is registered under compound_id.",
    )

    compound_id: str = Field(
        None, description="The unique compound identifier of the ligand."
    )

    dataset: str = Field(
        None,
        description='Dataset name from Fragalysis (name of structure).'
    )

    str_fn: str = Field(
        None,
        description='Filename of the PDB structure.'
    )

    sdf_fn: str = Field(
        None,
        description='Filename of the SDF file'
    )

class PDBStructure(Model):
    pdb_id: str = Field(
        None,
        description='PDB identification code.'
    )
    str_fn: str = Field(
        None,
        description='Filename of local PDB structure.'
    )

class EnantiomerPair(Model):
    active: ExperimentalCompoundData = Field(description='Active enantiomer.')
    inactive: ExperimentalCompoundData = Field(
        description='Inactive enantiomer.')

class EnantiomerPairList(Model):
    pairs: List[EnantiomerPair]


class DockingDataset(Model):
    class Config:
        allow_mutation=True
        arbitrary_types_allowed = True

    pkl_fn: str = Field(
        None,
        description='Filename of pickle containing info for docking results'
    )
    dir_path: str = Field(
        None,
        description='Filepath of dataset directory'
    )
    compound_ids: np.ndarray = Field(
        None,
        description='Numpy array of compound ids'
    )
    xtal_ids: np.ndarray = Field(
        None,
        description='Numpy array of structure ids'
    )
    res_ranks: np.ndarray = Field(
        None,
        description='Numpy array of sorted xtal_ids for each compound_id'
    )

    def read_pkl(self):
        self.compound_ids, self.xtal_ids, self.res_ranks = pkl.load(open(self.pkl_fn, 'rb'))
