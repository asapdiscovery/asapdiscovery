from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field

from asapdiscovery.data.backend.openeye import (
    combine_protein_ligand,
    load_openeye_pdb,
    oechem,
    save_openeye_pdb,
    split_openeye_mol,
)
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.schema.schema_base import ComplexBase, MoleculeFilter
from asapdiscovery.data.schema.target import Target

logger = logging.getLogger(__name__)


class Complex(ComplexBase):
    """
    Schema for a Complex, containing both a Target and Ligand
    In this case the Target field is required to be protein only

    """

    target: Target = Field(description="Target schema object")
    ligand: Ligand = Field(description="Ligand schema object")
    ligand_chain: str = Field(None, description="Chain ID of ligand in complex")

    # Overload from base class to check target and ligand individually
    def data_equal(self, other: Complex):
        return self.target.data_equal(other.target) and self.ligand.data_equal(
            other.ligand
        )

    @classmethod
    def from_oemol(
        cls,
        complex_mol: oechem.OEMol,
        target_chains=[],
        ligand_chain="",
        target_kwargs={},
        ligand_kwargs={},
    ) -> Complex:
        # Split molecule into parts using given chains
        mol_filter = MoleculeFilter(
            protein_chains=target_chains, ligand_chain=ligand_chain
        )
        split_dict = split_openeye_mol(complex_mol, mol_filter, keep_one_lig=True)

        # Create Target and Ligand objects
        target = Target.from_oemol(split_dict["prot"], **target_kwargs)
        ligand = Ligand.from_oemol(split_dict["lig"], **ligand_kwargs)

        return cls(
            target=target, ligand=ligand, ligand_chain=split_dict["keep_lig_chain"]
        )

    @classmethod
    def from_pdb(
        cls,
        pdb_file: str | Path,
        target_chains=[],
        ligand_chain="",
        target_kwargs={},
        ligand_kwargs={},
    ) -> Complex:
        # First load full complex molecule
        complex_mol = load_openeye_pdb(pdb_file)

        return cls.from_oemol(
            complex_mol=complex_mol,
            target_chains=target_chains,
            ligand_chain=ligand_chain,
            target_kwargs=target_kwargs,
            ligand_kwargs=ligand_kwargs,
        )

    def to_pdb(self, pdb_file: str | Path):
        save_openeye_pdb(self.to_combined_oemol(), pdb_file)

    def to_combined_oemol(self):
        """
        Combine the target and ligand into a single oemol
        """
        return combine_protein_ligand(
            self.target.to_oemol(), self.ligand.to_oemol(), lig_chain=self.ligand_chain
        )

    @property
    def hash(self):
        return f"{self.target.hash}+{self.ligand.fixed_inchikey}"


# Re-export for backward compatibility
from asapdiscovery.modeling.schema import PreppedComplex  # noqa: E402, F401
