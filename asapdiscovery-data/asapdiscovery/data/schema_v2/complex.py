from __future__ import annotations

from pathlib import Path
from typing import Any

from asapdiscovery.data.openeye import (
    combine_protein_ligand,
    load_openeye_pdb,
    save_openeye_pdb,
)
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

    # Overload from base class to check target and ligand individually
    def data_equal(self, other: Complex):
        return self.target.data_equal(other.target) and self.ligand.data_equal(
            other.ligand
        )

    @classmethod
    def from_pdb(
        cls,
        pdb_file: str | Path,
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
        lig_mol = split_dict["lig"]
        lig_mol.SetTitle(ligand_kwargs["compound_name"])
        ligand = Ligand.from_oemol(split_dict["lig"], **ligand_kwargs)

        return cls(target=target, ligand=ligand)

    def to_pdb(self, pdb_file: str | Path):
        lig_mol = self.ligand.to_oemol()
        target_mol = self.target.to_oemol()
        complex_mol = combine_protein_ligand(target_mol, lig_mol)

        save_openeye_pdb(complex_mol, pdb_file)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Complex):
            return NotImplemented

        # Just check that both Targets and Ligands are the same
        return (self.target == other.target) and (self.ligand == other.ligand)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)
