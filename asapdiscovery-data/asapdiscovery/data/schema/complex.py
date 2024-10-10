from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from asapdiscovery.data.backend.openeye import (
    combine_protein_ligand,
    load_openeye_design_unit,
    load_openeye_pdb,
    oechem,
    save_openeye_pdb,
)
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.schema.schema_base import DataModelAbstractBase
from asapdiscovery.data.schema.target import PreppedTarget, Target
from asapdiscovery.modeling.modeling import split_openeye_mol
from asapdiscovery.modeling.schema import MoleculeFilter
from pydantic import Field

logger = logging.getLogger(__name__)


class ComplexBase(DataModelAbstractBase):
    """
    Base class for complexes
    """

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ComplexBase):
            return NotImplemented

        # Just check that both Targets and Ligands are the same
        return (self.target == other.target) and (self.ligand == other.ligand)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @property
    def unique_name(self) -> str:
        """Create a unique name for the Complex, this is used in prep when generating folders to store results."""
        return f"{self.target.target_name}-{self.hash}"


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


class PreppedComplex(ComplexBase):
    """
    Schema for a Complex, containing both a PreppedTarget and Ligand
    In this case the PreppedTarget contains the protein and ligand.
    """

    target: PreppedTarget = Field(description="PreppedTarget schema object")
    ligand: Ligand = Field(description="Ligand schema object")

    # Overload from base class to check target and ligand individually
    def data_equal(self, other: PreppedComplex):
        return self.target.data_equal(other.target) and self.ligand.data_equal(
            other.ligand
        )

    @classmethod
    def from_oedu(
        cls, oedu: oechem.OEDesignUnit, target_kwargs={}, ligand_kwargs={}
    ) -> PreppedComplex:
        prepped_target = PreppedTarget.from_oedu(oedu, **target_kwargs)
        lig_oemol = oechem.OEMol()
        oedu.GetLigand(lig_oemol)
        return cls(
            target=prepped_target,
            ligand=Ligand.from_oemol(lig_oemol, **ligand_kwargs),
        )

    @classmethod
    def from_oedu_file(cls, oedu_file: str | Path, **kwargs) -> PreppedComplex:
        oedu = load_openeye_design_unit(oedu_file)
        return cls.from_oedu(oedu=oedu, **kwargs)

    @classmethod
    def from_complex(cls, complex: Complex, prep_kwargs={}) -> PreppedComplex:
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
        # use local import here to avoid circular imports
        from asapdiscovery.modeling.protein_prep import ProteinPrepper

        # overwrite ligand_chain with ligand_chain from complex if it exists
        prep_kwargs.pop("ligand_chain", None)
        prep_kwargs["ligand_chain"] = complex.ligand_chain
        prepped_complexs = ProteinPrepper(**prep_kwargs).prep(inputs=[complex])
        return prepped_complexs[0]

    @property
    def hash(self):
        # Using the target_hash instead hashing the OEDU bytes because prepping is stochastic
        return f"{self.target.target_hash}+{self.ligand.fixed_inchikey}"
