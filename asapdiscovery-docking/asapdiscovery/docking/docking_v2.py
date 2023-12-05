"""
Defines docking base schema.
"""

import abc
import logging
from pathlib import Path
from typing import Literal, Optional, Union

import dask
from asapdiscovery.data.dask_utils import actualise_dask_delayed_iterable
from asapdiscovery.data.openeye import combine_protein_ligand, oechem, save_openeye_pdb
from asapdiscovery.data.schema_v2.complex import PreppedComplex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.pairs import CompoundStructurePair
from asapdiscovery.data.schema_v2.sets import MultiStructureBase
from asapdiscovery.modeling.modeling import split_openeye_design_unit
from pydantic import BaseModel, Field, PositiveFloat

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

    def unique_name(self):
        return f"{self.complex.unique_name()}_{self.ligand.compound_name}-{self.ligand.fixed_inchikey}"


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
    def _dock(
        self, inputs: list[DockingInputPair], output_dir: Union[str, Path]
    ) -> list["DockingResult"]:
        ...

    def dock(
        self,
        inputs: list[DockingInputPair],
        output_dir: Union[str, Path],
        use_dask: bool = False,
        dask_client=None,
    ) -> Union[list[dask.delayed], list["DockingResult"]]:
        if not Path(output_dir).exists():
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        if use_dask:
            delayed_outputs = []
            for inp in inputs:
                out = dask.delayed(self._dock)(inputs=[inp], output_dir=output_dir)
                delayed_outputs.append(out[0])  # flatten
            outputs = actualise_dask_delayed_iterable(
                delayed_outputs, dask_client=dask_client, errors="skip"
            )
        else:
            outputs = self._dock(inputs=inputs, output_dir=output_dir)
        # filter out None values
        outputs = [o for o in outputs if o is not None]
        return outputs

    @staticmethod
    def write_docking_files(
        docking_results: list["DockingResult"], output_dir: Union[str, Path]
    ):
        """
        Write docking results to files in output_dir

        Parameters
        ----------
        docking_results : list[DockingResult]
            List of DockingResults
        output_dir : Union[str, Path]
            Output directory

        Raises
        ------
        """
        output_dir = Path(output_dir)

        # write out the docked poses and info
        for result in docking_results:
            result._write_docking_files(result, output_dir)

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]:
        ...

    @staticmethod
    def _write_docking_files(result: "DockingResult", output_dir: Union[str, Path]):
        output_pref = result.unique_name()
        compound_dir = output_dir / output_pref
        compound_dir.mkdir(parents=True, exist_ok=True)
        output_sdf_file = compound_dir / "docked.sdf"
        output_pdb_file = compound_dir / "docked_complex.pdb"
        output_json_file = compound_dir / "docking_result.json"
        result.posed_ligand.to_sdf(output_sdf_file)
        combined_oemol = result.to_posed_oemol()
        save_openeye_pdb(combined_oemol, output_pdb_file)
        result.to_json_file(output_json_file)


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

    def to_json_file(self, file: str | Path):
        with open(file, "w") as f:
            f.write(self.json(indent=2))

    @classmethod
    def from_json_file(cls, file: str | Path) -> "DockingResult":
        with open(file, "r") as f:
            return cls.parse_raw(f.read())

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

    def unique_name(self):
        return self.input_pair.unique_name()

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
