from pathlib import Path
from shutil import rmtree
from typing import Optional

from asapdiscovery.data.metadata.resources import master_structures
from asapdiscovery.data.operators.deduplicator import LigandDeDuplicator
from asapdiscovery.data.operators.selectors.mcs_selector import MCSSelector
from asapdiscovery.data.readers.meta_ligand_factory import MetaLigandFactory
from asapdiscovery.data.readers.meta_structure_factory import MetaStructureFactory
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.services.aws.cloudfront import CloudFront
from asapdiscovery.data.services.aws.s3 import S3
from asapdiscovery.data.services.postera.manifold_artifacts import (
    ArtifactType,
    ManifoldArtifactUploader,
)
from asapdiscovery.data.services.postera.manifold_data_validation import (
    rename_output_columns_for_manifold,
)
from asapdiscovery.data.services.postera.molecule_set import MoleculeSetAPI
from asapdiscovery.data.services.postera.postera_uploader import PosteraUploader
from asapdiscovery.data.services.services_config import (
    CloudfrontSettings,
    PosteraSettings,
    S3Settings,
)
from asapdiscovery.data.util.dask_utils import BackendType, make_dask_client_meta
from asapdiscovery.data.util.logging import FileLogger
from asapdiscovery.data.util.utils import check_empty_dataframe
from asapdiscovery.dataviz.html_viz import ColorMethod, HTMLVisualizer
from asapdiscovery.docking.docking import write_results_to_multi_sdf
from asapdiscovery.docking.docking_data_validation import DockingResultCols
from asapdiscovery.docking.openeye import POSITDocker
from asapdiscovery.docking.scorer import (
    ChemGauss4Scorer,
    FINTScorer,
    MetaScorer,
    MLModelScorer,
)
from asapdiscovery.genetics.fitness import target_has_fitness_data
from asapdiscovery.ml.models import ASAPMLModelRegistry
from asapdiscovery.modeling.protein_prep import ProteinPrepper
from asapdiscovery.workflows.docking_workflows.workflows import (
    PosteraDockingWorkflowInputs,
)
from pydantic import Field, PositiveInt, validator


class LargeScaleDockingInputs(PosteraDockingWorkflowInputs):
    """
    Schema for inputs to large scale docking

    Parameters
    ----------
    n_select : int, optional
        Number of targets to dock each ligand against, sorted by MCS
    top_n : int, optional
        Number of docking results to return, ordered by docking score
    posit_confidence_cutoff : float, optional
        POSIT confidence cutoff used to filter docking results
    use_omega : bool
        Whether to use omega confomer enumeration in docking, warning: more expensive
    allow_posit_retries : bool
        Whether to allow retries in docking with varying settings, warning: more expensive
    ml_scorers : ModelType, optional
        The name of the ml scorers to use.
    logname : str, optional
        Name of the log file.
    """

    top_n: PositiveInt = Field(
        500, description="Number of docking results to return, ordered by docking score"
    )

    posit_confidence_cutoff: float = Field(
        0.7,
        le=1.0,
        ge=0.0,
        description="POSIT confidence cutoff used to filter docking results",
    )

    use_omega: bool = Field(
        False,
        description="Whether to use omega confomer enumeration in docking, warning: more expensive",
    )

    allow_posit_retries: bool = Field(
        False,
        description="Whether to allow retries in docking with varying settings, warning: more expensive",
    )

    ml_scorers: Optional[list[str]] = Field(
        None, description="The name of the ml scorers to use"
    )

    @classmethod
    @validator("ml_scorers")
    def ml_scorers_must_be_valid(cls, v):
        """
        Validate that the ml scorers are valid
        """
        if v is not None:
            for ml_scorer in v:
                if ml_scorer not in ASAPMLModelRegistry.get_implemented_model_types():
                    raise ValueError(
                        f"ML scorer {ml_scorer} not valid, must be one of {ASAPMLModelRegistry.get_implemented_model_types()}"
                    )
        return v


def large_scale_docking_workflow(inputs: LargeScaleDockingInputs):
    """
    Run large scale docking on a set of ligands, against multiple targets

    Parameters
    ----------
    inputs : LargeScaleDockingInputs
        Inputs to large scale docking

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
        logfile="large-scale-docking.log",
        stdout=True,
        level=inputs.loglevel,
    ).getLogger()

    if new_directory:
        logger.info(f"Writing to / overwriting output directory: {output_dir}")
    else:
        logger.info(f"Writing to existing output directory: {output_dir}")

    logger.info(f"Running large scale docking with inputs: {inputs}")
    logger.info(f"Dumping input schema to {output_dir / 'inputs.json'}")

    # dump config to json file
    inputs.to_json_file(output_dir / "large_scale_docking_inputs.json")

    if inputs.use_dask:
        dask_client = make_dask_client_meta(
            inputs.dask_type,
            loglevel=inputs.loglevel,
        )
    else:
        dask_client = None

    # make a directory to store intermediate CSV results
    data_intermediates = Path(output_dir / "data_intermediates")
    data_intermediates.mkdir(exist_ok=True)

    if inputs.postera_upload:
        postera_settings = PosteraSettings()
        logger.info("Postera settings loaded")
        logger.info("Postera upload specified, checking for AWS credentials")
        aws_s3_settings = S3Settings()
        aws_cloudfront_settings = CloudfrontSettings()
        logger.info("AWS S3 and CloudFront credentials found")

    # read ligands
    ligand_factory = MetaLigandFactory(
        postera=inputs.postera,
        postera_molset_name=inputs.postera_molset_name,
        ligand_file=inputs.ligands,
    )
    query_ligands = ligand_factory.load()

    # read structures
    structure_factory = MetaStructureFactory(
        structure_dir=inputs.structure_dir,
        fragalysis_dir=inputs.fragalysis_dir,
        pdb_file=inputs.pdb_file,
    )
    complexes = structure_factory.load(
        use_dask=inputs.use_dask,
        dask_failure_mode=inputs.dask_failure_mode,
        dask_client=dask_client,
    )

    n_query_ligands = len(query_ligands)
    logger.info(f"Loaded {n_query_ligands} query ligands")

    # deduplicate ligands
    logger.info("Deduplicating by Inchikey")
    query_ligands = LigandDeDuplicator().deduplicate(query_ligands)
    n_query_ligands = len(query_ligands)
    logger.info(f"Deduplicated to {n_query_ligands} query ligands")

    n_complexes = len(complexes)
    logger.info(f"Loaded {n_complexes} complexes")

    # TODO: hide detail of canonical structure
    logger.info("Using canonical structure")
    align_struct = master_structures[inputs.target]

    ref_complex = Complex.from_pdb(
        align_struct,
        target_kwargs={"target_name": "ref"},
        ligand_kwargs={"compound_name": "ref_ligand"},
    )

    # prep complexes, possibly loading in from cache
    logger.info("Prepping complexes")
    prepper = ProteinPrepper(
        cache_dir=inputs.cache_dir,
        align=ref_complex,
        ref_chain="A",
        active_site_chain="A",
    )

    prepped_complexes = prepper.prep(
        complexes,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        dask_failure_mode=inputs.dask_failure_mode,
        cache_dir=inputs.cache_dir,
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
    # TODO: fix, see issue 560
    logger.info("Selecting pairs for docking based on MCS")
    selector = MCSSelector()
    pairs = selector.select(
        query_ligands,
        prepped_complexes,
        n_select=inputs.n_select,
    )

    n_pairs = len(pairs)
    logger.info(f"Selected {n_pairs} pairs for docking")

    del prepped_complexes

    # dock pairs, possibly reading and writing from a local cache of completed work
    logger.info("Running docking on selected pairs")
    docker = POSITDocker(
        use_omega=inputs.use_omega, allow_retries=inputs.allow_posit_retries
    )
    results = docker.dock(
        pairs,
        output_dir=output_dir / "docking_results",
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        dask_failure_mode=inputs.dask_failure_mode,
        return_for_disk_backend=True,
    )

    # NOTE: We use disk based dask backend here because the docking results are large and can cause memory issues
    # and thrashing with data transfer between workers and the scheduler, all the following operations are then marked
    # as using disk based dask backend

    n_results = len(results)
    logger.info(f"Docked {n_results} pairs successfully")
    if n_results == 0:
        raise ValueError("No docking results generated, exiting")
    del pairs

    if inputs.write_final_sdf:
        logger.info("Writing final docked poses to SDF file")
        write_results_to_multi_sdf(
            output_dir / "docking_results.sdf",
            results,
            backend=BackendType.DISK,
            reconstruct_cls=docker.result_cls,
        )

    # add chemgauss4 scorer
    scorers = [ChemGauss4Scorer()]

    if target_has_fitness_data(inputs.target):
        logger.info("Target has fitness data, adding FINT scorer")
        scorers.append(FINTScorer(target=inputs.target))

    # load ml scorers
    if inputs.ml_scorers:
        for ml_scorer in inputs.ml_scorers:
            logger.info(f"Loading ml scorer: {ml_scorer}")
            scorer = MLModelScorer.from_latest_by_target_and_type(
                inputs.target, ml_scorer
            )
            if scorer:
                scorers.append(scorer)

    # score results using multiple scoring functions
    logger.info("Scoring docking results")
    scorer = MetaScorer(scorers=scorers)

    logger.info("Running scoring")
    scores_df = scorer.score(
        results,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        dask_failure_mode=inputs.dask_failure_mode,
        return_df=True,
        backend=BackendType.DISK,
        reconstruct_cls=docker.result_cls,
    )

    scores_df.to_csv(data_intermediates / "docking_scores_raw.csv", index=False)

    # run html visualiser to get web-ready vis of docked poses
    logger.info("Running HTML visualiser for docked poses")
    html_ouptut_dir = output_dir / "poses"
    html_visualizer = HTMLVisualizer(
        color_method=ColorMethod.subpockets,
        target=inputs.target,
        output_dir=html_ouptut_dir,
    )
    pose_visualizatons = html_visualizer.visualize(
        results,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        dask_failure_mode=inputs.dask_failure_mode,
        backend=BackendType.DISK,
        reconstruct_cls=docker.result_cls,
    )
    # rename visualisations target id column to POSIT structure tag so we can join
    pose_visualizatons.rename(
        columns={
            DockingResultCols.TARGET_ID.value: DockingResultCols.DOCKING_STRUCTURE_POSIT.value
        },
        inplace=True,
    )

    # join the two dataframes on ligand_id, target_id and smiles
    scores_df = scores_df.merge(
        pose_visualizatons,
        on=[
            DockingResultCols.LIGAND_ID.value,
            DockingResultCols.DOCKING_STRUCTURE_POSIT.value,
            DockingResultCols.SMILES.value,
        ],
        how="outer",  # preserves rows where there is no visualisation
    )

    # run html viz of target fitness to get web-ready vis of docked poses
    if target_has_fitness_data(inputs.target):
        logger.info("Running fitness HTML visualiser")
        html_fitness_output_dir = output_dir / "fitness"
        html_fitness_visualizer = HTMLVisualizer(
            color_method=ColorMethod.fitness,
            target=inputs.target,
            output_dir=html_fitness_output_dir,
        )
        fitness_visualizations = html_fitness_visualizer.visualize(
            results,
            use_dask=inputs.use_dask,
            dask_client=dask_client,
            dask_failure_mode=inputs.dask_failure_mode,
            backend=BackendType.DISK,
            reconstruct_cls=docker.result_cls,
        )

        # duplicate target id column so we can join
        fitness_visualizations[
            DockingResultCols.DOCKING_STRUCTURE_POSIT.value
        ] = fitness_visualizations[DockingResultCols.TARGET_ID.value]

        # join the two dataframes on ligand_id, target_id and smiles
        scores_df = scores_df.merge(
            fitness_visualizations,
            on=[
                DockingResultCols.LIGAND_ID.value,
                DockingResultCols.DOCKING_STRUCTURE_POSIT.value,
                DockingResultCols.SMILES.value,
            ],
            how="outer",  # preserves rows where there is no fitness visualisation
        )
    else:
        logger.info(
            f"Target {inputs.target} does not have fitness data, skipping fitness visualisation"
        )

    logger.info("Filtering docking results")
    # filter for POSIT probability > 0.7
    scores_df = scores_df[
        scores_df[DockingResultCols.DOCKING_CONFIDENCE_POSIT.value]
        > inputs.posit_confidence_cutoff
    ]

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

    # then order by chemgauss4 score
    scores_df = scores_df.sort_values(
        DockingResultCols.DOCKING_SCORE_POSIT.value, ascending=True
    )
    scores_df.to_csv(
        data_intermediates / "docking_scores_filtered_sorted.csv", index=False
    )

    # remove duplicates that are the same compound docked to different structures
    scores_df = scores_df.drop_duplicates(
        subset=[DockingResultCols.SMILES.value], keep="first"
    )

    n_duplicate_filtered = len(scores_df)
    logger.info(
        f"Filtered to {n_duplicate_filtered} / {n_clash_filtered} docking results by duplicate ligand filter"
    )

    # set hit flag to False
    scores_df[DockingResultCols.DOCKING_HIT.value] = False

    # set top n hits to True
    scores_df.loc[
        scores_df.index[: inputs.top_n], DockingResultCols.DOCKING_HIT.value
    ] = True  # noqa: E712

    hits_df = scores_df[  # noqa: E712
        scores_df[DockingResultCols.DOCKING_HIT.value] == True  # noqa: E712
    ]

    n_top_n_filtered = len(hits_df)
    logger.info(
        f"Filtered to {n_top_n_filtered} / {n_duplicate_filtered} docking results by top n filter"
    )

    check_empty_dataframe(
        hits_df,
        logger=logger,
        fail="raise",
        tag="scores",
        message=f"No docking results passed the top {inputs.top_n} filter, no hits",
    )

    hits_df.to_csv(
        data_intermediates
        / f"docking_scores_filtered_sorted_top_{inputs.top_n}_hits.csv",
        index=False,
    )

    # rename columns for manifold
    logger.info("Renaming columns for manifold")

    # keep everything not just hits
    result_df = rename_output_columns_for_manifold(
        scores_df,
        inputs.target,
        [DockingResultCols],
        manifold_validate=True,
        drop_non_output=True,
        allow=[
            DockingResultCols.LIGAND_ID.value,
            DockingResultCols.HTML_PATH_POSE.value,
            DockingResultCols.HTML_PATH_FITNESS.value,
        ],
    )

    result_df.to_csv(output_dir / "docking_results_final.csv", index=False)

    if inputs.postera_upload:
        logger.info("Uploading results to Postera")

        postera_uploader = PosteraUploader(
            settings=PosteraSettings(), molecule_set_name=inputs.postera_molset_name
        )
        # push the results to PostEra, making a new molecule set if necessary
        manifold_data, molset_name, made_new_molset = postera_uploader.push(result_df)

        combined = postera_uploader.join_with_manifold_data(
            result_df,
            manifold_data,
            DockingResultCols.SMILES.value,
            DockingResultCols.LIGAND_ID.value,
            drop_no_uuid=True,
        )

        if made_new_molset:
            logger.info(f"Made new molecule set with name: {molset_name}")
        else:
            molset_name = inputs.postera_molset_name

        logger.info("Uploading artifacts to PostEra")

        # make an uploader for the poses and upload them
        artifact_columns = [
            DockingResultCols.HTML_PATH_POSE.value,
        ]
        artifact_types = [
            ArtifactType.DOCKING_POSE_POSIT,
        ]

        if target_has_fitness_data(inputs.target):
            artifact_columns.append(DockingResultCols.HTML_PATH_FITNESS.value)
            artifact_types.append(ArtifactType.DOCKING_POSE_FITNESS_POSIT)

        # upload artifacts to S3 and link them to postera
        uploader = ManifoldArtifactUploader(
            target=inputs.target,
            molecule_dataframe=combined,
            molecule_set_name=molset_name,
            bucket_name=aws_s3_settings.BUCKET_NAME,
            artifact_types=artifact_types,
            artifact_columns=artifact_columns,
            moleculeset_api=MoleculeSetAPI.from_settings(postera_settings),
            s3=S3.from_settings(aws_s3_settings),
            cloudfront=CloudFront.from_settings(aws_cloudfront_settings),
        )
        uploader.upload_artifacts()
