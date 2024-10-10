import abc
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import dask
import yaml
from asapdiscovery.data.backend.openeye import oechem
from asapdiscovery.data.schema.complex import Complex, PreppedComplex
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.schema.target import PreppedTarget
from asapdiscovery.data.util.dask_utils import (
    FailureMode,
    actualise_dask_delayed_iterable,
)
from asapdiscovery.data.util.stringenum import StringEnum
from asapdiscovery.data.util.utils import seqres_to_res_list
from asapdiscovery.modeling.modeling import (
    make_design_unit,
    mutate_residues,
    split_openeye_design_unit,
    spruce_protein,
    superpose_molecule,
)
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from distributed import Client

logger = logging.getLogger(__name__)


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
    def _prep(self, inputs: list[Complex]) -> list[PreppedComplex]: ...

    @staticmethod
    def _gather_new_tasks(
        complex_to_prep: list[Complex], cached_complexs: list[PreppedComplex]
    ) -> tuple[list[Complex], list[PreppedComplex]]:
        """
        For a set of complexs we want to prep gather a list of tasks to do removing complexs that have already
        been prepped and are in the cache.
        Parameters
        ----------
        complex_to_prep: The list of complexs we want to perform prep on.
        cached_complexs: The list of PreppedComplexs found in the cache which can be reused.

        Returns
        -------
            A tuple of two lists, the first contains the complexs which should be prepped and the second contains
            the PreppedComplex from the cache which should be reused.
        """
        cached_by_hash = {comp.hash: comp for comp in cached_complexs}
        # gather outputs which are in the cache
        cached_outputs = [
            cached_by_hash[inp.hash]
            for inp in complex_to_prep
            if inp.hash in cached_by_hash
        ]
        if cached_outputs:
            to_prep = [inp for inp in complex_to_prep if inp.hash not in cached_by_hash]
        else:
            to_prep = complex_to_prep

        return to_prep, cached_outputs

    def prep(
        self,
        inputs: list[Complex],
        use_dask: bool = False,
        dask_client: Optional["Client"] = None,
        failure_mode: FailureMode = FailureMode.SKIP,
        cache_dir: Optional[str] = None,
        use_only_cache: bool = False,
    ) -> list[PreppedComplex]:
        """
        Prepare the list of input receptor ligand complexs re-using any found in the cache.
        Parameters
        ----------
        inputs: The list of complexs to prepare.
        use_dask: If dask should be used to distribute the jobs.
        dask_client: The dask client that should be used to submit the jobs.
        failure_mode: The failure mode for dask. Can be 'raise' or 'skip'.
        cache_dir: The directory of previously cached PreppedComplexs which can be reused.

        Note
        ----
            Newly prepared structures are not cached automatically, call `cache` to store the results.

        Returns
        -------
            A list of prepared complexes.
        """
        all_outputs = []

        if cache_dir is not None:
            # make cache if it doesn't exist
            Path(cache_dir).mkdir(exist_ok=True, parents=True)
            cached_complexs = ProteinPrepperBase.load_cache(cache_dir=cache_dir)
            # workout what we can reuse
            if cached_complexs:
                logger.info(
                    f"Loaded {len(cached_complexs)} cached structures from: {cache_dir}."
                )
                # reduce the number of tasks using any possible cached structures
                inputs, cached_outputs = ProteinPrepperBase._gather_new_tasks(
                    complex_to_prep=inputs, cached_complexs=cached_complexs
                )
                if use_only_cache:
                    if inputs:
                        logger.warning(
                            f"Disregarding {len(inputs)} structures which could not be found in the cache."
                        )
                        inputs = None

                if cached_outputs:
                    logger.info(
                        f"Matched {len(cached_outputs)} cached structures which will be reused."
                    )
                    all_outputs.extend(cached_outputs)

        # check if we have something to run
        if inputs:
            logger.info(f"Prepping {len(inputs)} complexes")

            if use_dask:
                delayed_outputs = []
                for inp in inputs:
                    out = dask.delayed(self._prep)(inputs=[inp])
                    delayed_outputs.append(out[0])  # flatten
                outputs = actualise_dask_delayed_iterable(
                    delayed_outputs, dask_client, errors=failure_mode
                )  # skip here as some complexes may fail for various reasons
            else:
                outputs = self._prep(inputs=inputs, failure_mode=failure_mode)

            outputs = [o for o in outputs if o is not None]
            # save the newly calculated outputs
            all_outputs.extend(outputs)

        if len(all_outputs) == 0:
            raise ValueError(
                "No complexes were successfully prepped, likely that nothing was passed in, cache was mis-specified or cache was empty."
            )
        return all_outputs

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]: ...

    @staticmethod
    def cache(
        prepped_complexes: list[PreppedComplex],
        cache_dir: Union[str, Path],
    ) -> None:
        """
        Cache the list of PreppedComplex in its own folder. Each is saved as a JSON, oedu PDB and ligand SDF for vis.

        Args:
        """
        cache_dir = Path(cache_dir)
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)

        for pc in prepped_complexes:
            # create a folder for the complex data if its not already present
            complex_folder = cache_dir.joinpath(pc.unique_name)
            if not complex_folder.exists():
                complex_folder.mkdir(parents=True, exist_ok=True)
                pc.to_json_file(
                    complex_folder.joinpath(pc.target.target_name + ".json")
                )
                pc.target.to_oedu_file(
                    complex_folder.joinpath(pc.target.target_name + ".oedu")
                )
                pc.target.to_pdb_file(
                    complex_folder.joinpath(pc.target.target_name + ".pdb")
                )
                pc.ligand.to_sdf(
                    complex_folder.joinpath(pc.ligand.compound_name + ".sdf")
                )

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
        None, description="Chain ID to align to in reference structure"
    )
    active_site_chain: Optional[str] = Field(
        None,
        description="Active site chain ID to align to ref_chain in reference structure",
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

    def _prep(self, inputs: list[Complex], failure_mode="skip") -> list[PreppedComplex]:
        """
        Prepares a series of proteins for docking using OESpruce.
        """
        prepped_complexes = []
        for complex_target in inputs:

            logger.debug(
                f"Prepping complex: {complex_target.target.target_name} - {complex_target.ligand.compound_name}"
            )
            try:
                # load protein
                prot = complex_target.to_combined_oemol()

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
                    if "SEQRES" not in seqres_dict:
                        raise ValueError("No SEQRES found in YAML")
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
                    ids=complex_target.target.ids,
                    target_name=complex_target.target.target_name,
                    ligand_chain=complex_target.ligand_chain,
                    target_hash=complex_target.target.hash,
                )
                # we need the ligand at the new translated coordinates
                translated_oemol, _, _ = split_openeye_design_unit(du=du)
                translated_lig = Ligand.from_oemol(
                    translated_oemol, **complex_target.ligand.dict(exclude={"data"})
                )
                pc = PreppedComplex(target=prepped_target, ligand=translated_lig)
                pc.target.crystal_symmetry = complex_target.target.crystal_symmetry

                prepped_complexes.append(pc)

            except Exception as e:
                if failure_mode == "skip":
                    logger.error(
                        f"Failed to prep complex: {complex_target.target.target_name} - {e}"
                    )
                elif failure_mode == "raise":
                    raise e
                else:
                    raise ValueError(
                        f"Unknown error mode: {failure_mode}, must be 'skip' or 'raise'"
                    )

        return prepped_complexes

    def provenance(self) -> dict[str, str]:
        return {
            "prepper_type": self.prepper_type,
            "oechem": oechem.OEChemGetVersion(),
            "oespruce": oechem.OESpruceGetVersion(),
        }


class LigandTransferProteinPrepper(ProteinPrepper):
    """
    Protein prepper class that uses OESpruce to prepare a protein for docking.
    Creates a design unit by
    1) first prepping the protein,
    2) aligning it to the reference complex,
    3) then copying the ligand from the reference to the prepped protein.
    """

    prepper_type: Literal["ProteinPrepper"] = Field(
        "LigandTransferProteinPrepper", description="The type of prepper to use"
    )

    reference_complexes: list[Complex] = Field(
        ..., description="A list of reference complexes to transfer ligands from."
    )

    ref_chain: Optional[str] = Field("A", description="Reference chain ID to align to.")

    active_site_chain: Optional[str] = Field(
        "A", description="Chain ID to align to reference."
    )
    seqres_yaml: Optional[Path] = Field(
        None, description="Path to seqres yaml to mutate to."
    )
    loop_db: Optional[Path] = Field(
        None, description="Path to loop database to use for prepping"
    )

    def _prep(self, inputs: list[Complex], failure_mode="skip") -> list[PreppedComplex]:
        """
        Prepares a series of proteins for docking using OESpruce.
        """
        prepped_complexes = []
        for complex in inputs:
            logger.debug(f"Prepping {complex.target.target_name}")
            # load protein
            prot = complex.target.to_oemol()

            # mutate residues
            if self.seqres_yaml:
                with open(self.seqres_yaml) as f:
                    seqres_dict = yaml.safe_load(f)
                if "SEQRES" not in seqres_dict:
                    raise ValueError("No SEQRES found in YAML")
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

            # For each reference complex, align and transfer the ligand to the prepped protein
            logger.debug(
                f"Prepping with ligands from {len(self.reference_complexes)} reference complexes"
            )
            for complex_ref in self.reference_complexes:
                logger.debug(f"Reference complex: {complex_ref.target.target_name}")
                aligned, _ = superpose_molecule(
                    complex_ref.to_combined_oemol(),
                    spruced,
                    self.ref_chain,
                    self.active_site_chain,
                )

                ligand = complex_ref.ligand.to_oemol()

                from asapdiscovery.modeling.modeling import make_du_from_new_lig

                success, du = make_du_from_new_lig(
                    aligned,
                    ligand,
                )
                if not success:
                    warnings.warn(
                        f"Failed to make design unit for target {complex.target.target_name} and complex {complex_ref.unique_name}."
                    )
                    continue

                from asapdiscovery.data.backend.openeye import oedocking

                success = oedocking.OEMakeReceptor(du)

                if not success:
                    warnings.warn(
                        f"Made design unit, but failed to make receptor for target {complex.target.target_name} "
                        f"and complex {complex.unique_name}."
                    )
                    continue

                prepped_target = PreppedTarget.from_oedu(
                    du,
                    ids=complex.target.ids,
                    target_name=complex.target.target_name,
                    ligand_chain=self.active_site_chain,
                    target_hash=complex_ref.hash,
                )
                # we need the ligand at the new translated coordinates
                translated_oemol, _, _ = split_openeye_design_unit(du=du)
                translated_lig = Ligand.from_oemol(
                    translated_oemol, **complex_ref.ligand.dict(exclude={"data"})
                )
                pc = PreppedComplex(target=prepped_target, ligand=translated_lig)
                prepped_complexes.append(pc)

        return prepped_complexes
