"""
A test-oriented docking workflow for testing the docking pipeline.
Removes all the additional layers in the other workflows and adds some features to make running cross-docking easier
"""
import logging
from pathlib import Path
from shutil import rmtree
from typing import Optional

from asapdiscovery.data.dask_utils import (
    DaskType,
    dask_cluster_from_type,
    set_dask_config,
)
from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.postera.manifold_data_validation import (
    TargetTags,
    rename_output_columns_for_manifold,
)
from asapdiscovery.data.postera.postera_factory import PosteraFactory
from asapdiscovery.data.postera.postera_uploader import PosteraUploader
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.fragalysis import FragalysisFactory
from asapdiscovery.data.schema_v2.ligand import write_ligands_to_multi_sdf
from asapdiscovery.data.schema_v2.molfile import MolFileFactory
from asapdiscovery.data.schema_v2.structure_dir import StructureDirFactory
from asapdiscovery.data.selectors.mcs_selector import MCSSelector
from asapdiscovery.data.services_config import PosteraSettings
from asapdiscovery.data.utils import check_empty_dataframe
from asapdiscovery.docking.docking_data_validation import (
    DockingResultColsV2 as DockingResultCols,
)
from asapdiscovery.docking.openeye import POSIT_METHOD, POSIT_RELAX_MODE, POSITDocker
from asapdiscovery.docking.scorer_v2 import ChemGauss4Scorer, MetaScorer, MLModelScorer
from asapdiscovery.docking.workflows.workflows import WorkflowInputsBase
from asapdiscovery.ml.models.ml_models import ASAPMLModelRegistry
from asapdiscovery.modeling.protein_prep_v2 import CacheType, ProteinPrepper
from distributed import Client
from pydantic import Field, PositiveInt, root_validator, validator


class CrossDockingWorkflowInputs(WorkflowInputsBase):
    logname: str = Field("cross_docking", description="Name of the log file.")

    # Copied from POSITDocker
    relax: POSIT_RELAX_MODE = Field(
        POSIT_RELAX_MODE.NONE,
        description="When to check for relaxation either, 'clash', 'all', 'none'",
    )
    posit_method: POSIT_METHOD = Field(
        POSIT_METHOD.ALL, description="POSIT method to use"
    )
    use_omega: bool = Field(True, description="Use omega to generate conformers")
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
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir()

    logger = FileLogger(
        inputs.logname, path=output_dir, stdout=True, level=inputs.loglevel
    ).getLogger()

    logger.info(f"Running cross docking with inputs: {inputs}")
    logger.info(f"Dumping input schema to {output_dir / 'inputs.json'}")

    inputs.to_json_file(output_dir / "large_scale_docking_inputs.json")

    if inputs.use_dask:
        set_dask_config()
        logger.info(f"Using dask for parallelism of type: {inputs.dask_type}")
        dask_cluster = dask_cluster_from_type(inputs.dask_type)

        if inputs.dask_type.is_lilac():
            logger.info("Lilac HPC config selected, setting adaptive scaling")
            dask_cluster.adapt(
                minimum=10,
                maximum=inputs.dask_cluster_max_workers,
                wait_count=10,
                interval="1m",
            )
            logger.info(f"Estimating {inputs.dask_cluster_n_workers} workers")
            dask_cluster.scale(inputs.dask_cluster_n_workers)

        dask_client = Client(dask_cluster)
        logger.info(f"Using dask client: {dask_client}")
        logger.info(f"Using dask cluster: {dask_cluster}")
        logger.info(f"Dask client dashboard: {dask_client.dashboard_link}")

    else:
        dask_client = None

    # make a directory to store intermediate CSV results
    data_intermediates = Path(output_dir / "data_intermediates")
    data_intermediates.mkdir(exist_ok=True)

    # load from file
    logger.info(f"Loading ligands from file: {inputs.filename}")
    molfile = MolFileFactory.from_file(inputs.filename)
    query_ligands = molfile.ligands

    # load complexes from a directory, from fragalysis or from a pdb file
    if inputs.structure_dir:
        logger.info(f"Loading structures from directory: {inputs.structure_dir}")
        structure_factory = StructureDirFactory.from_dir(inputs.structure_dir)
        complexes = structure_factory.load(
            use_dask=inputs.use_dask, dask_client=dask_client
        )
    elif inputs.fragalysis_dir:
        logger.info(f"Loading structures from fragalysis: {inputs.fragalysis_dir}")
        fragalysis = FragalysisFactory.from_dir(inputs.fragalysis_dir)
        complexes = fragalysis.load(use_dask=inputs.use_dask, dask_client=dask_client)

    elif inputs.pdb_file:
        logger.info(f"Loading structures from pdb: {inputs.pdb_file}")
        complex = Complex.from_pdb(
            inputs.pdb_file,
            target_kwargs={"target_name": inputs.pdb_file.stem},
            ligand_kwargs={"compound_name": f"{inputs.pdb_file.stem}_ligand"},
        )
        complexes = [complex]

    else:
        raise ValueError(
            "Must specify either fragalysis_dir, structure_dir or pdb_file"
        )

    n_query_ligands = len(query_ligands)
    logger.info(f"Loaded {n_query_ligands} query ligands")
    n_complexes = len(complexes)
    logger.info(f"Loaded {n_complexes} complexes")

    # prep complexes
    logger.info("Prepping complexes")
    prepper = ProteinPrepper(cache_dir=inputs.cache_dir)
    prepped_complexes = prepper.prep(
        complexes, use_dask=inputs.use_dask, dask_client=dask_client
    )
    del complexes

    n_prepped_complexes = len(prepped_complexes)
    logger.info(f"Prepped {n_prepped_complexes} complexes")

    if inputs.gen_cache:
        # cache prepped complexes
        cache_path = output_dir / inputs.gen_cache
        logger.info(f"Caching prepped complexes to {cache_path}")
        for cache_type in inputs.cache_type:
            prepper.cache(prepped_complexes, cache_path, type=cache_type)

    # define selector and select pairs
    # using dask here is too memory intensive as each worker needs a copy of all the complexes in memory
    # which are quite large themselves, is only effective for large numbers of ligands and small numbers of complexes
    logger.info("Selecting pairs for docking based on MCS")
    selector = MCSSelector()
    pairs = selector.select(
        query_ligands,
        prepped_complexes,
        n_select=inputs.n_select,
        use_dask=False,
        dask_client=None,
    )

    n_pairs = len(pairs)
    logger.info(f"Selected {n_pairs} pairs for docking")

    del prepped_complexes

    # dock pairs
    logger.info("Running docking on selected pairs")
    docker = POSITDocker(use_omega=inputs.use_omega, allow_retries=inputs.allow_retries)
    results = docker.dock(
        pairs,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
    )

    n_results = len(results)
    logger.info(f"Docked {n_results} pairs successfully")
    del pairs

    # write docking results
    logger.info("Writing docking results")
    POSITDocker.write_docking_files(results, output_dir / "docking_results")

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
        results, use_dask=inputs.use_dask, dask_client=dask_client, return_df=True
    )

    del results

    scores_df.to_csv(data_intermediates / "docking_scores_raw.csv", index=False)

    n_posit_filtered = len(scores_df)
    logger.info(
        f"Filtered to {n_posit_filtered} / {n_results} docking results by POSIT confidence"
    )

    check_empty_dataframe(
        scores_df,
        logger=logger,
        fail="raise",
        tag="scores",
        message="No docking results passed the POSIT confidence cutoff",
    )

    # filter out clashes (chemgauss4 score > 0)
    scores_df = scores_df[scores_df[DockingResultCols.DOCKING_SCORE_POSIT] <= 0]

    n_clash_filtered = len(scores_df)
    logger.info(
        f"Filtered to {n_clash_filtered} / {n_posit_filtered} docking results by clash filter"
    )

    check_empty_dataframe(
        scores_df,
        logger=logger,
        fail="raise",
        tag="scores",
        message="No docking results passed the clash filter",
    )

    # then order by chemgauss4 score and remove duplicates by ligand id
    scores_df = scores_df.sort_values(
        DockingResultCols.DOCKING_SCORE_POSIT.value, ascending=True
    )
    scores_df.to_csv(
        data_intermediates / "docking_scores_filtered_sorted.csv", index=False
    )

    scores_df = scores_df.drop_duplicates(subset=[DockingResultCols.LIGAND_ID.value])

    n_duplicate_filtered = len(scores_df)
    logger.info(
        f"Filtered to {n_duplicate_filtered} / {n_clash_filtered} docking results by duplicate ligand filter"
    )

    # take top n results
    scores_df = scores_df.head(inputs.top_n)

    n_top_n_filtered = len(scores_df)
    logger.info(
        f"Filtered to {n_top_n_filtered} / {n_duplicate_filtered} docking results by top n filter"
    )

    check_empty_dataframe(
        scores_df,
        logger=logger,
        fail="raise",
        tag="scores",
        message=f"No docking results passed the top {inputs.top_n} filter",
    )

    scores_df.to_csv(
        data_intermediates / f"docking_scores_filtered_sorted_top_{inputs.top_n}.csv",
        index=False,
    )

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
