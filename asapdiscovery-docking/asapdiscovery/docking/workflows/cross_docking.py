"""
A test-oriented docking workflow for testing the docking pipeline.
Removes all the additional layers in the other workflows and adds some features to make running cross-docking easier
"""

from pathlib import Path
from shutil import rmtree

<<<<<<< HEAD
from asapdiscovery.data.readers.molfile import MolFileFactory
from asapdiscovery.data.readers.structure_dir import StructureDirFactory
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.schema.ligand import write_ligands_to_multi_sdf
from asapdiscovery.data.services.fragalysis.fragalysis_reader import FragalysisFactory
from asapdiscovery.data.services.postera.manifold_data_validation import (
    rename_output_columns_for_manifold,
)
from asapdiscovery.data.structural.selectors.selector_list import StructureSelector
from asapdiscovery.data.util.dask_utils import dask_cluster_from_type, set_dask_config
from asapdiscovery.data.util.logging import FileLogger

=======
from asapdiscovery.data.dask_utils import make_dask_client_meta
from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.postera.manifold_data_validation import (
    rename_output_columns_for_manifold,
)
from asapdiscovery.data.schema_v2.ligand import write_ligands_to_multi_sdf
from asapdiscovery.data.schema_v2.meta_structure_factory import MetaStructureFactory
from asapdiscovery.data.schema_v2.molfile import MolFileFactory
from asapdiscovery.data.selectors.selector_list import StructureSelector

>>>>>>> upstream/main
from asapdiscovery.docking.docking import DockingInputMultiStructure
from asapdiscovery.docking.docking_data_validation import DockingResultCols
from asapdiscovery.docking.openeye import POSIT_METHOD, POSIT_RELAX_MODE, POSITDocker
from asapdiscovery.docking.scorer import ChemGauss4Scorer, MetaScorer
from asapdiscovery.docking.workflows.workflows import DockingWorkflowInputsBase
from asapdiscovery.modeling.protein_prep import ProteinPrepper
from pydantic import Field, PositiveInt


class CrossDockingWorkflowInputs(DockingWorkflowInputsBase):
    logname: str = Field("", description="Name of the log file.")

    structure_selector: StructureSelector = Field(
        StructureSelector.LEAVE_SIMILAR_OUT,
        description="Structure selector to use for docking",
    )
    multi_reference: bool = Field(
        False,
        description="Whether to use multi reference docking, in which the docking_method "
        "recieves a DockingInputMultiStructure object instead of a DockingInputPair object",
    )

    # Copied from POSITDocker
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


def cross_docking_workflow(inputs: CrossDockingWorkflowInputs):
    """
    Run cross docking on a set of ligands, against multiple targets

    Parameters
    ----------
    inputs : CrossDockingWorkflowInputs
        Inputs to cross docking

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
        logfile="cross-docking.log",
        stdout=True,
        level=inputs.loglevel,
    ).getLogger()

    if new_directory:
        logger.info(f"Writing to / overwriting output directory: {output_dir}")
    else:
        logger.info(f"Writing to existing output directory: {output_dir}")

    logger.info(f"Running cross docking with inputs: {inputs}")
    logger.info(f"Dumping input schema to {output_dir / 'inputs.json'}")

    inputs.to_json_file(output_dir / "cross_docking_inputs.json")

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

    # make a directory to store intermediate CSV results
    data_intermediates = Path(output_dir / "data_intermediates")
    data_intermediates.mkdir(exist_ok=True)

    # load from file
    logger.info(f"Loading ligands from file: {inputs.ligands}")
    molfile = MolFileFactory(filename=inputs.ligands)
    query_ligands = molfile.load()

    # read structures
    structure_factory = MetaStructureFactory(
        structure_dir=inputs.structure_dir,
        fragalysis_dir=inputs.fragalysis_dir,
        pdb_file=inputs.pdb_file,
        use_dask=inputs.use_dask,
        dask_failure_mode=inputs.dask_failure_mode,
        dask_client=dask_client,
    )
    complexes = structure_factory.load()

    n_query_ligands = len(query_ligands)
    logger.info(f"Loaded {n_query_ligands} query ligands")
    n_complexes = len(complexes)
    logger.info(f"Loaded {n_complexes} complexes")

    # prep complexes
    logger.info("Prepping complexes")
    prepper = ProteinPrepper(cache_dir=inputs.cache_dir)
    prepped_complexes = prepper.prep(
        complexes,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        dask_failure_mode=inputs.dask_failure_mode,
        cache_dir=inputs.cache_dir,
        use_only_cache=inputs.use_only_cache,
    )
    del complexes

    n_prepped_complexes = len(prepped_complexes)
    logger.info(f"Prepped {n_prepped_complexes} complexes")

    if inputs.save_to_cache and inputs.cache_dir is not None:
        logger.info(f"Writing prepped complexes to global cache {inputs.cache_dir}")
        prepper.cache(prepped_complexes, inputs.cache_dir)

    # define selector and select pairs
    # using dask here is too memory intensive as each worker needs a copy of all the complexes in memory
    # which are quite large themselves, is only effective for large numbers of ligands and small numbers of complexes
    logger.info("Selecting pairs for docking")
    # TODO: MCS takes an n_select arg but Pairwise does not...meaning we are losing that option the way this is written
    selector = inputs.structure_selector.selector_cls()
    pairs = selector.select(
        query_ligands,
        prepped_complexes,
    )

    n_pairs = len(pairs)
    logger.info(f"Selected {n_pairs} pairs for docking")

    if inputs.multi_reference:
        # Generate one-to-many objects for multi-reference docking
        logger.info("Generating one-to-many objects for multi-reference docking")
        sets = DockingInputMultiStructure.from_pairs(pairs)
        logger.info(f"Generated {len(sets)} ligand-protein sets for docking")
    else:
        sets = pairs

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
        sets,
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

    # rename columns for manifold
    logger.info("Renaming columns for manifold")
    result_df = rename_output_columns_for_manifold(
        scores_df,
        inputs.target,
        [DockingResultCols],
        manifold_validate=True,
        drop_non_output=True,
        allow=[DockingResultCols.LIGAND_ID.value],
    )

    result_df.to_csv(output_dir / "docking_results_final.csv", index=False)
