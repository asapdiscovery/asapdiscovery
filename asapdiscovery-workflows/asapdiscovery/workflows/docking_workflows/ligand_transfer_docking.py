"""
The ligand transfer docking workflow is a workflow that is used to dock a
ligand from one protein to another, related protein.

The workflow is as follows:
1. Load a set of apo protein targets
2. Load a set of ligands whose binding poses are known in the context of a reference protein
3. Align the apo proteins to the reference protein.
4. Transfer the ligand coordinates to the aligned apo proteins
5. Dock the transferred ligands to the apo proteins using the transferred coordinates as a reference
"""

from pathlib import Path
from shutil import rmtree

from asapdiscovery.data.operators.selectors.selector_list import StructureSelector
from asapdiscovery.data.readers.meta_structure_factory import MetaStructureFactory
from asapdiscovery.data.readers.molfile import MolFileFactory
from asapdiscovery.data.schema.ligand import write_ligands_to_multi_sdf
from asapdiscovery.data.services.postera.manifold_data_validation import (
    rename_output_columns_for_manifold,
)
from asapdiscovery.data.util.dask_utils import make_dask_client_meta
from asapdiscovery.data.util.logging import FileLogger
from asapdiscovery.docking.docking import DockingInputMultiStructure
from asapdiscovery.docking.docking_data_validation import DockingResultCols
from asapdiscovery.docking.openeye import POSIT_METHOD, POSIT_RELAX_MODE, POSITDocker
from asapdiscovery.docking.scorer import ChemGauss4Scorer, MetaScorer
from asapdiscovery.modeling.protein_prep import (
    LigandTransferProteinPrepper,
)
from asapdiscovery.workflows.docking_workflows.workflows import WorkflowInputsBase
from pydantic import Field, PositiveInt, root_validator
from typing import Optional, Union


class LigandTransferDockingWorkflowInputs(WorkflowInputsBase):
    target_structure_dir: Optional[Path] = Field(
        None,
        description="Path to a directory containing apo structures to transfer the ligands.",
    )
    target_pdb_file: Optional[Path] = Field(
        None, description="Path to a PDB file to prep and dock to."
    )

    target_fragalysis_dir: Optional[Path] = Field(
        None, description="Path to a directory containing a Fragalysis dump."
    )

    reference_complex_dir: Optional[Path] = Field(
        None, description="Path to a directory containing reference complexes."
    )
    reference_pdb_file: Optional[Path] = Field(
        None, description="Path to a PDB file to prep and dock to."
    )

    reference_fragalysis_dir: Optional[Path] = Field(
        None, description="Path to a directory containing a Fragalysis dump."
    )

    cache_dir: Optional[str] = Field(
        None, description="Path to a directory where a cache has been generated"
    )

    use_only_cache: bool = Field(
        False,
        description="Whether to only use the cached structures, otherwise try to prep uncached structures.",
    )

    save_to_cache: bool = Field(
        True,
        description="Generate a cache from structures prepped in this workflow run in this directory",
    )
    write_final_sdf: bool = Field(
        default=True,
        description="Whether to write the final docked poses to an SDF file.",
    )

    # Copied from LigandTransferProteinPrepper
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

    # Copied from CrossDockingWorkflowInputs
    relax: POSIT_RELAX_MODE = Field(
        POSIT_RELAX_MODE.NONE,
        description="When to check for relaxation either, 'clash', 'all', 'none'",
    )
    posit_method: POSIT_METHOD = Field(
        POSIT_METHOD.ALL, description="POSIT method to use"
    )
    use_omega: bool = Field(False, description="Use omega to generate conformers")
    omega_dense: bool = Field(False, description="Use dense conformer generation")
    num_poses: PositiveInt = Field(1, description="Number of poses to generate")
    allow_low_posit_prob: bool = Field(False, description="Allow low posit probability")
    low_posit_prob_thresh: float = Field(
        0.1,
        description="Minimum posit probability threshold if allow_low_posit_prob is False",
    )
    allow_final_clash: bool = Field(
        False, description="Allow clashing poses in last stage of docking"
    )
    allow_retries: bool = Field(
        True,
        description="Allow retries with different options if docking fails initially",
    )

    @root_validator
    @classmethod
    def check_inputs(cls, values):
        """
        Validate inputs
        """
        for _type in ["target", "reference"]:
            total = sum(
                [
                    bool(
                        values.get(f"{_type}_{_dir}")
                        for _dir in ["fragalysis_dir", "structure_dir", "pdb_file"]
                    )
                ]
            )

            # can only specify one of fragalysis dir, structure dir and PDB file
            if total != 1:
                raise ValueError(
                    f"Must specify exactly one of {_type}_fragalysis_dir, {_type}_structure_dir or {_type}_pdb_file"
                )

            return values


def ligand_transfer_docking_workflow(inputs: LigandTransferDockingWorkflowInputs):
    """
    Run ligand transfer docking on a set of ligands, against a set of targets

    Parameters
    ----------
    inputs : LigandTransferDockingWorkflowInputs
        Inputs to ligand transfer docking

    Returns
    -------
    None
    """
    output_dir = inputs.output_dir
    new_directory = True
    if output_dir.exists():
        if inputs.overwrite:
            rmtree(output_dir)
        else:
            new_directory = False

    # this won't overwrite the existing directory
    output_dir.mkdir(exist_ok=True, parents=True)

    logger = FileLogger(
        inputs.logname,  # default root logger so that dask logging is forwarded
        path=output_dir,
        logfile="ligand-transfer-docking.log",
        stdout=False,
        level=inputs.loglevel,
    ).getLogger()

    if new_directory:
        logger.info(f"Writing to / overwriting output directory: {output_dir}")
    else:
        logger.info(f"Writing to existing output directory: {output_dir}")

    logger.info(f"Running ligand-transfer docking with inputs: {inputs}")
    logger.info(f"Dumping input schema to {output_dir / 'inputs.json'}")
    inputs.to_json_file(output_dir / "ligand_transfer_docking_inputs.json")

    # make a directory to store intermediate CSV results
    data_intermediates = Path(output_dir / "data_intermediates")
    data_intermediates.mkdir(exist_ok=True)

    if inputs.use_dask:
        dask_client = make_dask_client_meta(
            inputs.dask_type,
            adaptive_min_workers=inputs.dask_cluster_n_workers,
            adaptive_max_workers=inputs.dask_cluster_max_workers,
            loglevel=inputs.loglevel,
            walltime=inputs.walltime,
        )
    else:
        dask_client = None

    # read structures
    ref_complex_factory = MetaStructureFactory(
        structure_dir=inputs.reference_complex_dir,
        fragalysis_dir=inputs.reference_fragalysis_dir,
        pdb_file=inputs.reference_pdb_file,
    )
    ref_complexes = ref_complex_factory.load(
        use_dask=inputs.use_dask,
        dask_failure_mode=inputs.dask_failure_mode,
        dask_client=dask_client,
    )

    # read structures
    target_factory = MetaStructureFactory(
        structure_dir=inputs.target_structure_dir,
        fragalysis_dir=inputs.target_fragalysis_dir,
        pdb_file=inputs.target_pdb_file,
    )
    targets = target_factory.load(
        use_dask=inputs.use_dask,
        dask_failure_mode=inputs.dask_failure_mode,
        dask_client=dask_client,
    )

    n_ref_complexes = len(ref_complexes)
    logger.info(f"Loaded {n_ref_complexes} reference complexes")
    n_targets = len(targets)
    logger.info(f"Loaded {n_targets} complexes")

    # prep complexes
    logger.info("Prepping complexes")
    prepper = LigandTransferProteinPrepper(
        reference_complexes=ref_complexes,
        ref_chain=inputs.ref_chain,
        active_site_chain=inputs.active_site_chain,
        seqres_yaml=inputs.seqres_yaml,
        loop_db=inputs.loop_db,
    )
    prepped_complexes = prepper.prep(
        targets,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        dask_failure_mode=inputs.dask_failure_mode,
        cache_dir=inputs.cache_dir,
        use_only_cache=inputs.use_only_cache,
    )
    del targets

    n_prepped_complexes = len(prepped_complexes)
    logger.info(f"Prepped {n_prepped_complexes} complexes")

    if inputs.save_to_cache and inputs.cache_dir is not None:
        logger.info(f"Writing prepped complexes to global cache {inputs.cache_dir}")
        prepper.cache(prepped_complexes, inputs.cache_dir)

    # Here the only thing that makes sense is the self docking selector
    selector = StructureSelector.SELF_DOCKING.selector_cls()
    ligands = [pc.ligand for pc in prepped_complexes]
    pairs = selector.select(ligands, prepped_complexes)

    n_pairs = len(pairs)
    logger.info(f"Selected {n_pairs} pairs for docking")
    del prepped_complexes

    # dock pairs
    logger.info("Running docking on selected pairs")
    docker = POSITDocker(
        relax=inputs.relax,
        posit_method=inputs.posit_method,
        use_omega=inputs.use_omega,
        omega_dense=inputs.omega_dense,
        num_poses=inputs.num_poses,
        allow_low_posit_prob=inputs.allow_low_posit_prob,
        low_posit_prob_thresh=inputs.low_posit_prob_thresh,
        allow_final_clash=inputs.allow_final_clash,
        allow_retries=inputs.allow_retries,
    )
    results = docker.dock(
        pairs,
        output_dir=output_dir / "docking_results",
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        dask_failure_mode=inputs.dask_failure_mode,
    )

    n_results = len(results)
    logger.info(f"Docked {n_results} pairs successfully")
    if n_results == 0:
        raise ValueError("No docking results generated, exiting")
    del pairs

    # add chemgauss4 scorer
    scorers = [ChemGauss4Scorer()]

    # score results
    logger.info("Scoring docking results")
    scorer = MetaScorer(scorers=scorers)

    if inputs.write_final_sdf:
        logger.info("Writing final docked poses to SDF file")
        write_ligands_to_multi_sdf(
            output_dir / "docking_results.sdf", [r.posed_ligand for r in results]
        )

    scores_df = scorer.score(
        results,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        return_df=True,
        dask_failure_mode=inputs.dask_failure_mode,
    )

    del results

    scores_df.to_csv(data_intermediates / "docking_scores_raw.csv", index=False)
