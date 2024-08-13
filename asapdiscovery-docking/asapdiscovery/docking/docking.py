"""
Defines docking base schema.
"""

import abc
import json
import logging
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
from asapdiscovery.data.backend.openeye import (
    combine_protein_ligand,
    oechem,
    save_openeye_pdb,
)
from asapdiscovery.data.schema.complex import Complex, PreppedComplex
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.schema.pairs import CompoundStructurePair
from asapdiscovery.data.schema.sets import MultiStructureBase
from asapdiscovery.data.schema.target import Target
from asapdiscovery.data.util.dask_utils import BackendType, FailureMode
from asapdiscovery.modeling.modeling import split_openeye_design_unit
from pydantic import BaseModel, Field, PositiveFloat

logger = logging.getLogger(__name__)


class DockingInputBase(BaseModel):
    """
    Base class for functionality all docking inputs should have.
    """

    @abc.abstractmethod
    def to_design_units(self) -> list[oechem.OEDesignUnit]: ...


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


class DockingInputMultiStructure(MultiStructureBase, DockingInputBase):
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
    ) -> list["DockingResult"]: ...

    def dock(
        self,
        inputs: list[DockingInputPair],
        output_dir: Optional[Union[str, Path]] = None,
        use_dask: bool = False,
        dask_client=None,
        failure_mode=FailureMode.SKIP,
        return_for_disk_backend: bool = False,
    ) -> list["DockingResult"]:
        """
        Run docking on a list of DockingInputPairs

        Parameters
        ----------
        inputs : list[DockingInputPair]
            List of DockingInputPairs
        output_dir : Optional[Union[str, Path]], optional
            Output directory, to write docking results to, by default None
            means no output files are written
        use_dask : bool, optional
            Whether to use dask, by default False
        dask_client : dask.distributed.Client, optional
            Dask client to use, by default None
        failure_mode : FailureMode, optional
            Dask failure mode, by default FailureMode.SKIP
        return_for_disk_backend : bool, optional
            Whether to return the results for disk backend, by default False

        Returns
        -------
        Union[list[DockingResult], list[Path]]]
            List of DockingResults or paths to DockingResult json files
        """
        # make output dir if it doesn't exist
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        outputs = self._dock(
            inputs=inputs,
            output_dir=output_dir,
            use_dask=use_dask,
            dask_client=dask_client,
            failure_mode=failure_mode,
            return_for_disk_backend=return_for_disk_backend,
        )

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
            result.write_docking_files(output_dir)

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]: ...


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
    pose_id: Optional[int] = Field(description="Nth returned pose from docking")
    provenance: dict[str, str] = Field(description="Provenance")

    def to_json_file(self, file: str | Path):
        with open(file, "w") as f:
            f.write(self.json(indent=2))

    @classmethod
    def from_json_file(cls, file: str | Path) -> "DockingResult":
        with open(file) as f:
            return cls.parse_raw(f.read())

    @abc.abstractmethod
    def _get_single_pose_results(self) -> list["DockingResult"]: ...

    def get_single_pose_results(self) -> list["DockingResult"]:
        return self._get_single_pose_results()

    def get_output(self) -> dict:
        """
        return a dictionary of some of the fields of the DockingResult
        """
        dct = self.dict()
        dct.pop("input_pair")
        dct.pop("posed_ligand")
        dct.pop("type")
        return dct

    @classmethod
    def from_json(cls, json_str):
        return cls.parse_obj(json.loads(json_str))

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
        param = self.input_pair.complex.target.crystal_symmetry
        if param is not None:
            p = oechem.OECrystalSymmetryParams(*param)
            oechem.OESetCrystalSymmetry(prot, p, True)
        return prot

    def to_posed_complex(self) -> Complex:
        """
        Return the complex from the original target
        Returns
        -------
        Complex
            Complex
        """
        prot = self.to_protein()
        lig = self.posed_ligand.to_oemol()
        tar = Target.from_oemol(
            prot,
            target_name=self.input_pair.complex.target.target_name,
            ids=self.input_pair.complex.target.ids,
        )
        lig = Ligand.from_oemol(lig, **self.input_pair.ligand.dict())
        return Complex(target=tar, ligand=lig)

    @property
    def unique_name(self):
        return self.input_pair.unique_name

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

    def write_docking_files(self, output_dir: Union[str, Path]):
        """
        Write docking files to output_dir

        Parameters
        ----------
        output_dir : Union[str, Path]
            Output directory
        """
        self._write_docking_files(self, output_dir)

    @staticmethod
    def _write_docking_files(result: "DockingResult", output_dir: Union[str, Path]):
        output_dir = Path(output_dir)
        output_pref = result.unique_name
        compound_dir = output_dir / f"{output_pref}_pose_{result.pose_id}"
        compound_dir.mkdir(parents=True, exist_ok=True)
        output_sdf_file = compound_dir / "docked.sdf"
        output_pdb_file = compound_dir / "docked_complex.pdb"
        output_json_file = compound_dir / "docking_result.json"
        result.posed_ligand.to_sdf(output_sdf_file)
        combined_oemol = result.to_posed_oemol()
        save_openeye_pdb(combined_oemol, output_pdb_file)
        result.provenance["on_disk_location"] = str(output_json_file.resolve())
        result.to_json_file(output_json_file)
        return output_json_file

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DockingResult):
            raise NotImplementedError

        # Just check that both Complexs and Ligands are the same
        return (
            (self.input_pair == other.input_pair)
            and (self.posed_ligand == other.posed_ligand)
            and (np.isclose(self.probability, other.probability))
        )

    def __neq__(self, other: Any) -> bool:
        return not self.__eq__(other)


def write_results_to_multi_sdf(
    sdf_file: Union[str, Path],
    results: Union[list[DockingResult], list[Path]],
    backend=BackendType.IN_MEMORY,
    reconstruct_cls=None,
):
    """
    Write a list of DockingResults to a single sdf file
    Can accept either a list of DockingResults or a list of paths to DockingResult json files
    depending on the backend used

    Parameters
    ----------
    results : Union[list[DockingResult], list[Path]]
        List of DockingResults or paths to DockingResult json files
    backend : BackendType, optional
        Backend to use, by default BackendType.IN_MEMORY
    reconstruct_cls : Optional[DockingResult], optional
        DockingResult class to use for disk backend, by default None

    Raises
    ------
    ValueError
        If backend is disk and no reconstruct_cls is provided
    """
    if backend == BackendType.DISK and not reconstruct_cls:
        raise ValueError("Must provide reconstruct_cls if using disk backend")

    for res in results:
        if backend == BackendType.IN_MEMORY:
            lig = res.posed_ligand
        elif backend == BackendType.DISK:
            lig = reconstruct_cls.from_json_file(res).posed_ligand
        else:
            raise ValueError(f"Unknown backend type {backend}")

        lig.to_sdf(sdf_file, allow_append=True)
