from pathlib import Path
from typing import Any, Union

from asapdiscovery.data.openeye import load_openeye_pdb
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.schema_base import DataModelAbstractBase
from asapdiscovery.data.schema_v2.target import Target
from asapdiscovery.modeling.modeling import split_openeye_mol
from asapdiscovery.modeling.schema import MoleculeFilter
from pydantic import Field


class Complex(DataModelAbstractBase):
    """
    Schema for a Complex, containing both a Target and Ligand
    """

    target: Target = Field(description="Target schema object")
    ligand: Ligand = Field(description="Ligand schema object")

    @classmethod
    def from_pdb(
        cls,
        pdb_file: Union[str, Path],
        target_chains=[],
        ligand_chain="",
        target_kwargs={},
        ligand_kwargs={},
    ):
        # First load full complex molecule
        complex_mol = load_openeye_pdb(pdb_file)

        # Split molecule into parts using given chains
        mol_filter = MoleculeFilter(
            protein_chains=target_chains, ligand_chain=ligand_chain
        )
        split_dict = split_openeye_mol(complex_mol, mol_filter)

        # Create Target and Ligand objects
        target = Target.from_oemol(split_dict["prot"], **target_kwargs)
        ligand = Ligand.from_oemol(split_dict["lig"], **ligand_kwargs)

        return cls(target=target, ligand=ligand)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Complex):
            return NotImplemented

        # Just check that both Targets and Ligands are the same
        return (self.target == other.target) and (self.ligand == other.ligand)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)
