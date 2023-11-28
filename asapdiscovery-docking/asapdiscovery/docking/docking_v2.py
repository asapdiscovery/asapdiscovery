"""
Defines docking base schema.
"""

import abc
from pathlib import Path
from typing import Literal, Optional, Union

import dask
from asapdiscovery.data.dask_utils import actualise_dask_delayed_iterable
from asapdiscovery.data.openeye import combine_protein_ligand, oechem, save_openeye_pdb
from asapdiscovery.data.schema_v2.complex import PreppedComplex
from asapdiscovery.data.schema_v2.ligand import Ligand, compound_names_unique
from asapdiscovery.data.schema_v2.pairs import CompoundStructurePair
from asapdiscovery.data.schema_v2.sets import MultiStructureBase
from asapdiscovery.modeling.modeling import split_openeye_design_unit
from pydantic import BaseModel, Field, PositiveFloat

import logging

logger = logging.getLogger(__name__)


class DockingInputBase(BaseModel):
    """
    Base class for functionality all docking inputs should have.
    """

    @abc.abstractmethod
    def to_design_units(self) -> list[oechem.OEDesignUnit]:
        ...


class DockingInputPair(CompoundStructurePair, DockingInputBase):
    """
    Schema for a DockingInputPair, containing both a PreppedComplex and Ligand
    This is designed to track a matched ligand and complex pair for investigation
    but with the complex prepped for docking, ie in OEDesignUnit format.
    """

    complex: PreppedComplex = Field(description="Target schema object")

    @classmethod
    def from_compound_structure_pair(
        cls, compound_structure_pair: CompoundStructurePair
    ) -> "DockingInputPair":
        prepped_complex = PreppedComplex.from_complex(compound_structure_pair.complex)
        return cls(complex=prepped_complex, ligand=compound_structure_pair.ligand)

    def to_design_units(self) -> list[oechem.OEDesignUnit]:
        return [self.complex.target.to_oedu()]


class DockingInputMultiStructure(MultiStructureBase):
    """
    Schema for one ligand to many possible reference structures.
    """

    ligand: Ligand = Field(description="Ligand schema object")
    complexes: list[PreppedComplex] = Field(description="List of reference structures")

    @classmethod
    def from_pairs(
        cls, pair_list: list[DockingInputPair]
    ) -> list["DockingInputMultiStructure"]:
        return cls._from_pairs(pair_list)

    def to_design_units(self) -> list[oechem.OEDesignUnit]:
        return [protein_complex.target.to_oedu() for protein_complex in self.complexes]


class DockingBase(BaseModel):
    """
    Base class for running docking
    """

    type: Literal["DockingBase"] = "DockingBase"

    @abc.abstractmethod
    def _dock(self, inputs: list[DockingInputPair]) -> list["DockingResult"]:
        ...

    def dock(
        self, inputs: list[DockingInputPair], use_dask: bool = False, dask_client=None
    ) -> Union[list[dask.delayed], list["DockingResult"]]:
        if use_dask:
            delayed_outputs = []
            for inp in inputs:
                out = dask.delayed(self._dock)(inputs=[inp])
                delayed_outputs.append(out[0])  # flatten
            outputs = actualise_dask_delayed_iterable(
                delayed_outputs, dask_client=dask_client, errors="skip"
            )
        else:
            outputs = self._dock(inputs=inputs)
        # filter out None values
        outputs = [o for o in outputs if o is not None]
        return outputs

    @staticmethod
    def write_docking_files(
        docking_results: list["DockingResult"], output_dir: Union[str, Path]
    ):
        """
        Write docking results to files in output_dir, directories will have the form:
        {target_name}_+_{ligand_name}/docked.sdf
        {target_name}_+_{ligand_name}/docked_complex.pdb

        Parameters
        ----------
        docking_results : list[DockingResult]
            List of DockingResults
        output_dir : Union[str, Path]
            Output directory

        Raises
        ------
        ValueError
            If compound names of input pair and posed ligand do not match

        """
        ligs = [docking_result.input_pair.ligand for docking_result in docking_results]
        names_unique = compound_names_unique(ligs)
        output_dir = Path(output_dir)
        # if names are not unique, we will use unknown_ligand_{i} as the ligand portion of directory
        # when writing files

        # write out the docked pose
        for i, result in enumerate(docking_results):
            if (
                not result.input_pair.ligand.compound_name
                == result.posed_ligand.compound_name
            ):
                raise ValueError(
                    "Compound names of input pair and posed ligand do not match"
                )
            if names_unique:
                output_pref = (
                    result.input_pair.complex.target.target_name
                    + "_+_"
                    + result.posed_ligand.compound_name
                )
            else:
                output_pref = (
                    result.input_pair.complex.target.target_name
                    + "_+_"
                    + f"unknown_ligand_{i}"
                )

            compound_dir = output_dir / output_pref
            compound_dir.mkdir(parents=True, exist_ok=True)
            output_sdf_file = compound_dir / "docked.sdf"
            output_pdb_file = compound_dir / "docked_complex.pdb"

            result.posed_ligand.to_sdf(output_sdf_file)

            combined_oemol = result.to_posed_oemol()
            save_openeye_pdb(combined_oemol, output_pdb_file)

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]:
        ...


class DockingResult(BaseModel):
    """
    Schema for a DockingResult, containing both a DockingInputPair used as input to the workflow
    and a Ligand object containing the docked pose.
    Also contains the probability of the docked pose if applicable.

    Parameters
    ----------
    input_pair : DockingInputPair
        Input pair
    posed_ligand : Ligand
        Posed ligand
    probability : float, optional
        Probability of the docked pose, by default None
    provenance : dict[str, str]
        Provenance information

    """

    type: Literal["DockingResult"] = "DockingResult"
    input_pair: DockingInputPair = Field(description="Input pair")
    posed_ligand: Ligand = Field(description="Posed ligand")
    probability: Optional[PositiveFloat] = Field(
        description="Probability"
    )  # not easy to get the probability from rescoring
    provenance: dict[str, str] = Field(description="Provenance")

    def get_output(self) -> dict:
        """
        return a dictionary of some of the fields of the DockingResult
        """
        dct = self.dict()
        dct.pop("input_pair")
        dct.pop("posed_ligand")
        dct.pop("type")
        return dct

    def to_posed_oemol(self) -> oechem.OEMol:
        """
        Combine the original target and posed ligand into a single oemol

        Returns
        -------
        oechem.OEMol
            Combined oemol
        """
        return combine_protein_ligand(self.to_protein(), self.posed_ligand.to_oemol())

    def to_protein(self) -> oechem.OEMol:
        """
        Return the protein from the original target

        Returns
        -------
        oechem.OEMol
            Protein oemol
        """
        _, prot, _ = split_openeye_design_unit(self.input_pair.complex.target.to_oedu())
        return prot

    def get_combined_id(self) -> str:
        """
        Get a unique ID for the DockingResult

        Returns
        -------
        str
            Unique ID
        """
        return (
            self.input_pair.complex.target.target_name
            + "_+_"
            + self.input_pair.ligand.compound_name
        )

    @staticmethod
    def make_df_from_docking_results(results: list["DockingResult"]):
        """
        Make a dataframe from a list of DockingResults

        Parameters
        ----------
        results : list[DockingResult]
            List of DockingResults

        Returns
        -------
        pd.DataFrame
            Dataframe of DockingResults
        """
        import pandas as pd

        return pd.DataFrame([r.get_output() for r in results])
