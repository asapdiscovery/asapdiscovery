import abc

from pathlib import Path
from typing import Literal, Optional, Union

import dask
import yaml
from asapdiscovery.data.dask_utils import actualise_dask_delayed_iterable
from asapdiscovery.data.enum import StringEnum
from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.schema_v2.complex import Complex, PreppedComplex
from asapdiscovery.data.schema_v2.target import PreppedTarget
from asapdiscovery.data.utils import seqres_to_res_list
from asapdiscovery.modeling.modeling import (
    make_design_unit,
    mutate_residues,
    spruce_protein,
    superpose_molecule,
)
from pydantic import BaseModel, Field, root_validator


class CacheType(StringEnum):
    """
    Enum for cache types.
    """

    DesignUnit = "DesignUnit"
    JSON = "JSON"


class ProteinPrepperBase(BaseModel):
    """
    Base class for protein preppers.
    """

    prepper_type: Literal["ProteinPrepperBase"] = Field(
        "ProteinPrepperBase", description="The type of prepper to use"
    )

    class Config:
        arbitrary_types_allowed = True

    @abc.abstractmethod
    def _prep(self) -> list[PreppedComplex]:
        ...

    def prep(
        self, inputs: list[Complex], use_dask: bool = False, dask_client=None
    ) -> list[PreppedComplex]:
        """
        Prepare the list of input receptor ligand complexs re-using any found in the cache.

        Parameters
        ----------
        inputs: The list of complexs to prepare.
        use_dask: If dask should be used to distribute the jobs.
        dask_client: The dask client that should be used to submit the jobs.

        Returns
        -------
            A list of prepared complexs.
        """
        if use_dask:
            delayed_outputs = []
            for inp in inputs:
                out = dask.delayed(self._prep)(inputs=[inp])
                delayed_outputs.append(out[0])  # flatten
            outputs = actualise_dask_delayed_iterable(
                delayed_outputs, dask_client, errors="skip"
            )  # skip here as some complexes may fail for various reasons
        else:
            outputs = self._prep(inputs=inputs)

        outputs = [o for o in outputs if o is not None]
        if len(outputs) == 0:
            raise ValueError(
                "No complexes were successfully prepped, likely that nothing was passed in, cache was mis-specified or cache was empty."
            )
        return outputs

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]:
        ...

    @staticmethod
    def cache(
        prepped_complexes: list[PreppedComplex],
        dir: Union[str, Path],
    ) -> None:
        """
        Cache the list of PreppedComplex in its own folder. Each is saved as a JSON, oedu PDB and ligand SDF for vis.
        """
        dir = Path(dir)
        if not dir.exists():
            dir.mkdir(parents=True)

        for pc in prepped_complexes:
            # create a folder for the complex data
            complex_folder = dir.joinpath(f"{pc.target.target_name}-{pc.hash()}")
            complex_folder.mkdir(parents=True, exist_ok=True)
            pc.to_json_file(complex_folder.joinpath(pc.target.target_name + ".json"))
            pc.target.to_oedu_file(
                complex_folder.joinpath(pc.target.target_name + ".oedu")
            )
            pc.target.to_pdb_file(
                complex_folder.joinpath(pc.target.target_name + ".pdb")
            )
            pc.ligand.to_sdf(complex_folder.joinpath(pc.ligand.compound_name + ".sdf"))

    @staticmethod
    def load_cache(
        cache_dir: Union[str, Path],
    ) -> list[PreppedComplex]:
        """
        Load a set of cached PreppedComplexes which can be reused.
        """
        if not (cache_dir := Path(cache_dir)).exists():
            raise ValueError(f"Cache directory {cache_dir} does not exist.")

        prepped_complexes = []
        for complex_file in cache_dir.rglob("*.json"):
            prepped_complexes.append(PreppedComplex.from_json_file(complex_file))

        return prepped_complexes


class ProteinPrepper(ProteinPrepperBase):
    """
    Protein prepper class that uses OESpruce to prepare a protein for docking.
    """

    prepper_type: Literal["ProteinPrepper"] = Field(
        "ProteinPrepper", description="The type of prepper to use"
    )

    align: Optional[Complex] = Field(
        None, description="Reference structure to align to."
    )
    ref_chain: Optional[str] = Field(
        None, description="Reference chain ID to align to."
    )
    active_site_chain: Optional[str] = Field(
        None, description="Chain ID to align to reference."
    )
    seqres_yaml: Optional[Path] = Field(
        None, description="Path to seqres yaml to mutate to."
    )
    loop_db: Optional[Path] = Field(
        None, description="Path to loop database to use for prepping"
    )
    oe_active_site_residue: Optional[str] = Field(
        None, description="OE formatted string of active site residue to use"
    )
    cache_dir: Optional[Path] = Field(
        None, description="Path to a directory where design units are cached"
    )

    fail_missing_cache: bool = Field(
        False, description="Whether to fail on missing files when loading from cache"
    )

    @root_validator
    @classmethod
    def _check_align_and_chain_info(cls, values):
        """
        Check that align and chain info is provided correctly.
        """
        align = values.get("align")
        ref_chain = values.get("ref_chain")
        active_site_chain = values.get("active_site_chain")
        if align and not ref_chain:
            raise ValueError("Must provide ref_chain if align is provided")
        if align and not active_site_chain:
            raise ValueError("Must provide active_site_chain if align is provided")
        return values

    def _prep(self, inputs: list[Complex]) -> list[PreppedComplex]:
        """
        Prepares a series of proteins for docking using OESpruce.
        """
        prepped_complexes = []
        if self.cache_dir:
            prepped_complexes = self.load_cache(
                complexes=inputs,
                cache_dir=self.cache_dir,
                fail_missing_cache=self.fail_missing_cache,
            )
        else:
            for complex in inputs:
                # load protein
                prot = complex.to_combined_oemol()

                if self.align:
                    prot, _ = superpose_molecule(
                        self.align.to_combined_oemol(),
                        prot,
                        self.ref_chain,
                        self.active_site_chain,
                    )

                # mutate residues
                if self.seqres_yaml:
                    with open(self.seqres_yaml) as f:
                        seqres_dict = yaml.safe_load(f)
                    seqres = seqres_dict["SEQRES"]
                    res_list = seqres_to_res_list(seqres)
                    prot = mutate_residues(prot, res_list, place_h=True)
                    protein_sequence = " ".join(res_list)
                else:
                    seqres = None
                    protein_sequence = None

                # spruce protein
                success, spruce_error_message, spruced = spruce_protein(
                    initial_prot=prot,
                    protein_sequence=protein_sequence,
                    loop_db=self.loop_db,
                )

                if not success:
                    raise ValueError(
                        f"Prep failed, with error message: {spruce_error_message}"
                    )

                success, du = make_design_unit(
                    spruced,
                    site_residue=self.oe_active_site_residue,
                    protein_sequence=protein_sequence,
                )
                if not success:
                    raise ValueError("Failed to make design unit.")

                prepped_target = PreppedTarget.from_oedu(
                    du,
                    ids=complex.target.ids,
                    target_name=complex.target.target_name,
                    ligand_chain=complex.ligand_chain,
                    target_hash=complex.target.hash(),
                )
                pc = PreppedComplex(target=prepped_target, ligand=complex.ligand)
                prepped_complexes.append(pc)

        return prepped_complexes

    def provenance(self) -> dict[str, str]:
        return {
            "prepper_type": self.prepper_type,
            "oechem": oechem.OEChemGetVersion(),
            "oespruce": oechem.OESpruceGetVersion(),
        }
