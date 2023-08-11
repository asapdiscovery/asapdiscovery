import abc
from pydantic import BaseModel, Field, validator
from asapdiscovery.data.schema_v2.ligand import Ligand
from enum import Enum

from typing import List

class StateExpanderType(str, Enum):
    STEREO = "STEREO"
    TAUTOMER = "TAUTOMER"
    PROTOMER = "PROTOMER"
    


class StateExpanderBase(BaseModel):

    input_ligands: List[Ligand] = Field(..., description="The ligand to be expanded")
    expander_type: StateExpanderType = Field(..., description="The type of expander to use")


    @abc.abstractmethod
    def _expand(self) -> List["StateExpansion"]:
        ...
    
    def expand(self) -> List["StateExpansion"]:
        return self._expand()
    
    
        

class StateExpansion(BaseModel):
    parent: Ligand = Field(..., description="The parent ligand")
    children: List[Ligand] = Field(..., description="The children ligands resulting from expansion")
    expander: StateExpanderBase = Field(..., description="The expander used to generate the children")

    @property
    def n_expanded_states(self) -> int:
        return len(self.children)
    
    @property
    def parent_smiles(self) -> str:
        return self.parent.smiles

    @staticmethod
    def flatten_children(expansions: List["StateExpansion"]) -> List[Ligand]:
        return [child for expansion in expansions for child in expansion.children]
    
    @staticmethod
    def flatten_parents(expansions: List["StateExpansion"]) -> List[Ligand]:
        return [expansion.parent for expansion in expansions]

