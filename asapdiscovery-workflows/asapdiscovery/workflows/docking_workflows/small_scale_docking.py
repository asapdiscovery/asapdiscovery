from pathlib import Path
from shutil import rmtree

from asapdiscovery.data.metadata.resources import master_structures
from asapdiscovery.data.operators.deduplicator import LigandDeDuplicator
from asapdiscovery.data.operators.selectors.mcs_selector import RascalMCESSelector
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
    TargetProteinMap,
    map_output_col_to_manifold_tag,
    rename_output_columns_for_manifold,
)
from asapdiscovery.data.services.postera.molecule_set import MoleculeSetAPI
from asapdiscovery.data.services.postera.postera_uploader import PosteraUploader
from asapdiscovery.data.services.services_config import (
    CloudfrontSettings,
    PosteraSettings,
    S3Settings,
)
from asapdiscovery.data.util.dask_utils import (
    BackendType,
    DaskType,
    make_dask_client_meta,
)
from asapdiscovery.data.util.logging import FileLogger
from asapdiscovery.data.util.utils import check_empty_dataframe
from asapdiscovery.dataviz.gif_viz import GIFVisualizer
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
from asapdiscovery.ml.models import ASAPMLModelRegistry
from asapdiscovery.modeling.protein_prep import ProteinPrepper
from asapdiscovery.simulation.simulate import OpenMMPlatform, VanillaMDSimulator
from asapdiscovery.spectrum.fitness import target_has_fitness_data
from asapdiscovery.workflows.docking_workflows.workflows import (
    PosteraDockingWorkflowInputs,
)
from pydantic import Field, PositiveInt


class SmallScaleDockingInputs(PosteraDockingWorkflowInputs):
    """
    Schema for inputs to small scale docking

    Parameters
    ----------
    posit_confidence_cutoff : float, optional
        POSIT confidence cutoff used to filter docking results
    use_omega : bool
        Whether to use omega for conformer generation prior to docking
    allow_retries : bool
        Whether to allow retries for docking failures
    n_select : PositiveInt
        Number of targets to dock each ligand against.
    ml_score: bool, optional
        Whether to use ML scoring in the docking pipeline
    md : bool, optional
        Whether to run MD on the docked poses
    md_steps : PositiveInt, optional
        Number of MD steps to run
    md_report_interval : PositiveInt, optional
        MD report interval for writing to disk
    md_openmm_platform : OpenMMPlatform, optional
        OpenMM platform to use for MD
    logname : str, optional
        Name of the log file.
    """

    posit_confidence_cutoff: float = Field(
        0.1,
        le=1.0,
        ge=0.0,
        description="POSIT confidence cutoff used to filter docking results",
    )

    use_omega: bool = Field(
        False,
        description="Whether to use omega for conformer generation prior to docking",
    )

    allow_retries: bool = Field(
        True, description="Whether to allow retries for docking failures"
    )

    n_select: PositiveInt = Field(
        3, description="Number of targets to dock each ligand against."
    )

    ml_score: bool = Field(
        True, description="Whether to use ML scoring in the docking pipeline"
    )

    allow_dask_cuda: bool = Field(
        True,
        description="Whether to allow regenerating dask cuda cluster when in local mode",
    )

    md: bool = Field(False, description="Whether to run MD on the docked poses")
    md_steps: PositiveInt = Field(2500000, description="Number of MD steps to run")
    md_report_interval: PositiveInt = Field(
        1250, description="MD report interval for writing to disk"
    )
    md_openmm_platform: OpenMMPlatform = Field(
        OpenMMPlatform.Fastest, description="OpenMM platform to use for MD"
    )


def small_scale_docking_workflow(inputs: SmallScaleDockingInputs):
    """
    Run small scale docking on a set of ligands, against multiple targets

    Parameters
    ----------
    inputs : SmallScaleDockingInputs
        Inputs to small scale docking

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
        logfile="small-scale-docking.log",
        stdout=True,
        level=inputs.loglevel,
    ).getLogger()

    if new_directory:
        logger.info(f"Writing to / overwriting output directory: {output_dir}")
    else:
        logger.info(f"Writing to existing output directory: {output_dir}")

    logger.info(f"Running small scale docking with inputs: {inputs}")
    logger.info(f"Dumping input schema to {output_dir / 'inputs.json'}")

    # dump config to json file
    inputs.to_json_file(output_dir / "small_scale_docking_inputs.json")

    if inputs.use_dask:
        dask_client = make_dask_client_meta(
            inputs.dask_type,
            loglevel=inputs.loglevel,
            n_workers=inputs.dask_n_workers,
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
        use_dask=inputs.use_dask,
        failure_mode=inputs.failure_mode,
        dask_client=dask_client,
    )
    complexes = structure_factory.load(
        use_dask=inputs.use_dask,
        failure_mode=inputs.failure_mode,
        dask_client=dask_client,
    )

    n_query_ligands = len(query_ligands)
    logger.info(f"Loaded {n_query_ligands} query ligands")
    logger.info("Deduplicating by Inchikey")
    query_ligands = LigandDeDuplicator().deduplicate(query_ligands)
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

    # prep complexes
    logger.info("Prepping complexes")
    prepper = ProteinPrepper(
        cache_dir=inputs.cache_dir,
        align=ref_complex,
        ref_chain=inputs.ref_chain,
        active_site_chain=inputs.active_site_chain,
    )
    prepped_complexes = prepper.prep(
        complexes,
        cache_dir=inputs.cache_dir,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        failure_mode=inputs.failure_mode,
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
    logger.info("Selecting pairs for docking based on MCS")
    selector = RascalMCESSelector(
        similarity_threshold=0.4
    )  # better attempt to find the MCS than the default 0.7
    pairs = selector.select(
        query_ligands,
        prepped_complexes,
        n_select=inputs.n_select,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        failure_mode=inputs.failure_mode,
    )

    n_pairs = len(pairs)
    logger.info(f"Selected {n_pairs} pairs for docking")

    del prepped_complexes

    # dock pairs
    logger.info("Running docking on selected pairs")
    docker = POSITDocker(
        use_omega=inputs.use_omega,
        allow_retries=inputs.allow_retries,
        last_ditch_fred=True,
    )
    results = docker.dock(
        pairs,
        output_dir=output_dir / "docking_results",
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        failure_mode=inputs.failure_mode,
        return_for_disk_backend=True,
    )

    n_results = len(results)
    logger.info(f"Docked {n_results} pairs successfully")
    if n_results == 0:
        raise ValueError("No docking results generated, exiting")
    del pairs

    # add chemgauss4 scorer
    scorers = [ChemGauss4Scorer()]

    if target_has_fitness_data(inputs.target):
        logger.info("Target has fitness data, adding FINT scorer")
        scorers.append(FINTScorer(target=inputs.target))

    # load ml scorers
    if inputs.ml_score:
        # check which endpoints are availabe for the target
        models = ASAPMLModelRegistry.reccomend_models_for_target(inputs.target)
        for model in models:
            logger.info(
                f"Adding ML scorer for target {inputs.target} with model {model.name}"
            )

        ml_scorers = MLModelScorer.load_model_specs(models=models)
        scorers.extend(ml_scorers)

    if inputs.write_final_sdf:
        logger.info("Writing final docked poses to SDF file")
        write_results_to_multi_sdf(
            output_dir / "docking_results.sdf",
            results,
            backend=BackendType.DISK,
            reconstruct_cls=docker.result_cls,
        )
    # score results with multiple scoring functions
    logger.info("Scoring docking results")
    scorer = MetaScorer(scorers=scorers)
    scores_df = scorer.score(
        results,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        failure_mode=inputs.failure_mode,
        return_df=True,
        backend=BackendType.DISK,
        reconstruct_cls=docker.result_cls,
        return_for_disk_backend=True,
    )

    scores_df.to_csv(data_intermediates / "docking_scores_raw.csv", index=False)

    logger.info("Filtering docking results")
    # filter for POSIT probability
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

    logger.info("Running HTML visualiser for docked poses")
    html_ouptut_dir = output_dir / "poses"
    html_visualizer = HTMLVisualizer(
        color_method=ColorMethod.subpockets,
        target=inputs.target,
        output_dir=html_ouptut_dir,
        ref_chain=inputs.ref_chain,
        active_site_chain=inputs.ref_chain,
        backend=BackendType.DISK,
        reconstruct_cls=docker.result_cls,
    )
    pose_visualizatons = html_visualizer.visualize(
        results,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        failure_mode=inputs.failure_mode,
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
    combined_df = scores_df.merge(
        pose_visualizatons,
        on=[
            DockingResultCols.LIGAND_ID.value,
            DockingResultCols.DOCKING_STRUCTURE_POSIT.value,
            DockingResultCols.SMILES.value,
        ],
        how="outer",
    )

    if target_has_fitness_data(inputs.target):
        logger.info("Running fitness HTML visualiser")
        html_fitness_output_dir = output_dir / "fitness"
        html_fitness_visualizer = HTMLVisualizer(
            color_method=ColorMethod.fitness,
            target=inputs.target,
            output_dir=html_fitness_output_dir,
            ref_chain=inputs.ref_chain,
            active_site_chain=inputs.ref_chain,
        )
        fitness_visualizations = html_fitness_visualizer.visualize(
            results,
            use_dask=inputs.use_dask,
            dask_client=dask_client,
            failure_mode=inputs.failure_mode,
            backend=BackendType.DISK,
            reconstruct_cls=docker.result_cls,
        )

        # duplicate target id column so we can join
        fitness_visualizations[DockingResultCols.DOCKING_STRUCTURE_POSIT.value] = (
            fitness_visualizations[DockingResultCols.TARGET_ID.value]
        )

        # join the two dataframes on ligand_id, target_id and smiles
        combined_df = combined_df.merge(
            fitness_visualizations,
            on=[
                DockingResultCols.LIGAND_ID.value,
                DockingResultCols.DOCKING_STRUCTURE_POSIT.value,
                DockingResultCols.SMILES.value,
            ],
            how="outer",
        )
    else:
        logger.info(
            f"Not running fitness HTML visualiser because {inputs.target} does not have fitness data"
        )

    # filter out clashes (chemgauss4 score > 0)
    combined_df = combined_df[combined_df[DockingResultCols.DOCKING_SCORE_POSIT] <= 0]

    n_clash_filtered = len(combined_df)
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
    combined_df = combined_df.sort_values(
        DockingResultCols.DOCKING_SCORE_POSIT.value, ascending=True
    )
    combined_df.to_csv(
        data_intermediates / "docking_scores_filtered_sorted.csv", index=False
    )

    # remove duplicates that are the same compound docked to different structures
    combined_df = combined_df.drop_duplicates(
        subset=[DockingResultCols.SMILES.value], keep="first"
    )

    # re-extract the filtered input results
    results = combined_df["input"].tolist()

    n_duplicate_filtered = len(combined_df)
    logger.info(
        f"Filtered to {n_duplicate_filtered} / {n_clash_filtered} docking results by duplicate ligand filter"
    )

    if inputs.md:
        local_cpu_client_gpu_override = False
        if (
            (inputs.allow_dask_cuda)
            and (inputs.dask_type == DaskType.LOCAL)
            and (inputs.use_dask)
        ):
            logger.info(
                "Using local CPU dask cluster, and MD has been requested, replacing with a GPU cluster"
            )
            dask_client = make_dask_client_meta(
                DaskType.LOCAL_GPU, loglevel=inputs.loglevel
            )
            local_cpu_client_gpu_override = True

        md_output_dir = output_dir / "md"

        # capsid simulations need a CA rmsd restraint to hold the capsid together
        if TargetProteinMap[inputs.target] == "Capsid":
            logger.info("Adding CA RMSD restraint to capsid simulation")
            rmsd_restraint = True
            rmsd_restraint_type = "CA"
        else:
            rmsd_restraint = False
            rmsd_restraint_type = None

        md_simulator = VanillaMDSimulator(
            output_dir=md_output_dir,
            openmm_platform=inputs.md_openmm_platform,
            num_steps=inputs.md_steps,
            reporting_interval=inputs.md_report_interval,
            rmsd_restraint=rmsd_restraint,
            rmsd_restraint_type=rmsd_restraint_type,
        )
        simulation_results = md_simulator.simulate(
            results,
            use_dask=inputs.use_dask,
            dask_client=dask_client,
            failure_mode=inputs.failure_mode,
            backend=BackendType.DISK,
            reconstruct_cls=docker.result_cls,
        )

        if len(simulation_results) == 0:
            raise ValueError("No MD simulation results generated, exiting")

        if local_cpu_client_gpu_override and inputs.use_dask:
            dask_client = make_dask_client_meta(DaskType.LOCAL)

        gif_output_dir = output_dir / "gifs"

        # take the last ns, accounting for possible low number of frames
        start_frame = max(md_simulator.n_frames - md_simulator.frames_per_ns, 1)

        logger.info(f"Using start frame {start_frame} for GIFs")
        gif_maker = GIFVisualizer(
            output_dir=gif_output_dir,
            target=inputs.target,
            frames_per_ns=md_simulator.frames_per_ns,
            start=start_frame,
        )
        gifs = gif_maker.visualize(
            simulation_results,
            use_dask=inputs.use_dask,
            dask_client=dask_client,
            failure_mode=inputs.failure_mode,
        )
        gifs.to_csv(data_intermediates / "md_gifs.csv", index=False)
        # duplicate target id column so we can join
        gifs[DockingResultCols.DOCKING_STRUCTURE_POSIT.value] = gifs[
            DockingResultCols.TARGET_ID.value
        ]

        # join the two dataframes on ligand_id, target_id and smiles
        combined_df = combined_df.merge(
            gifs,
            on=[
                DockingResultCols.LIGAND_ID.value,
                DockingResultCols.DOCKING_STRUCTURE_POSIT.value,
                DockingResultCols.SMILES.value,
            ],
            how="outer",
        )

    # rename columns for manifold
    logger.info("Renaming columns for manifold")

    result_df = rename_output_columns_for_manifold(
        combined_df,
        inputs.target,
        [DockingResultCols],
        manifold_validate=True,
        drop_non_output=True,
        allow=[
            DockingResultCols.HTML_PATH_POSE.value,
            DockingResultCols.HTML_PATH_FITNESS.value,
            DockingResultCols.GIF_PATH.value,
            DockingResultCols.LIGAND_ID.value,
        ],
    )

    result_df.to_csv(output_dir / "docking_results_final.csv", index=False)

    if inputs.postera_upload:
        logger.info("Uploading numerical results to Postera")
        posit_score_tag = map_output_col_to_manifold_tag(
            DockingResultCols, inputs.target
        )[DockingResultCols.DOCKING_SCORE_POSIT.value]

        postera_uploader = PosteraUploader(
            settings=PosteraSettings(),
            molecule_set_name=inputs.postera_molset_name,
        )

        # push the results to PostEra, making a new molecule set if necessary
        manifold_data, molset_name, made_new_molset = postera_uploader.push(
            result_df, sort_column=posit_score_tag, sort_ascending=True
        )

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

        if inputs.md:
            artifact_columns.append(DockingResultCols.GIF_PATH.value)
            artifact_types.append(ArtifactType.MD_POSE)

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
        uploader.upload_artifacts(sort_column=posit_score_tag, sort_ascending=True)
