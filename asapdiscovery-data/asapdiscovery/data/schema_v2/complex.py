from __future__ import annotations

from pathlib import Path
from typing import Any


from asapdiscovery.data.openeye import (
    load_openeye_pdb,
    combine_protein_ligand,
)
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.schema_base import DataModelAbstractBase
from asapdiscovery.data.schema_v2.target import Target, PreppedTarget
from asapdiscovery.modeling.modeling import split_openeye_mol
from asapdiscovery.modeling.protein_prep_v2 import ProteinPrepper
from asapdiscovery.modeling.schema import MoleculeFilter
from pydantic import Field


class ComplexBase(DataModelAbstractBase):
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ComplexBase):
            return NotImplemented

        # Just check that both Targets and Ligands are the same
        return (self.target == other.target) and (self.ligand == other.ligand)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


class Complex(ComplexBase):
    """
    Schema for a Complex, containing both a Target and Ligand
    In this case the Target field is required to be protein only

    """

    target: Target = Field(description="Target schema object")
    ligand: Ligand = Field(description="Ligand schema object")
    ligand_chain: str = Field(..., description="Chain ID of ligand in complex")

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
    ) -> "Complex":
        # First load full complex molecule
        complex_mol = load_openeye_pdb(pdb_file)

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

    def to_combined_oemol(self):
        """
        Combine the target and ligand into a single oemol
        """
        return combine_protein_ligand(
            self.target.to_oemol(), self.ligand.to_oemol(), lig_chain=self.ligand_chain
        )


class PreppedComplex(ComplexBase):
    """
    Schema for a Complex, containing both a PreppedTarget and Ligand
    In this case the PreppedTarget contains the protein and ligand.
    """

    target: PreppedTarget = Field(description="PreppedTarget schema object")
    ligand: Ligand = Field(description="Ligand schema object")
    ligand_chain: str = Field(..., description="Chain ID of ligand in complex")

    # Overload from base class to check target and ligand individually
    def data_equal(self, other: PreppedComplex):
        return self.target.data_equal(other.target) and self.ligand.data_equal(
            other.ligand
        )

    @classmethod
    def from_complex(cls, complex: Complex, prep_kwargs={}) -> "PreppedComplex":
        """
        Create a PreppedComplex from a Complex by running ProteinPrepper
        on the combined oemol of the complex

        Parameters
        ----------
        complex : Complex
            Complex to create PreppedComplex from
        prep_kwargs : dict
            Keyword arguments to pass to ProteinPrepper

        Returns
        -------
        PreppedComplex
            PreppedComplex object
        """
        # overwrite ligand_chain with ligand_chain from complex if it exists
        prep_kwargs.pop("ligand_chain", None)
        prep_kwargs["ligand_chain"] = complex.ligand_chain
        oedu = ProteinPrepper(**prep_kwargs).prep(complex.to_combined_oemol())
        # copy over ids from complex
        prepped_target = PreppedTarget.from_oedu(
            oedu,
            ids=complex.target.ids,
            target_name=complex.target.target_name,
            ligand_chain=complex.ligand_chain,
        )
        return cls(target=prepped_target, ligand=complex.ligand)
