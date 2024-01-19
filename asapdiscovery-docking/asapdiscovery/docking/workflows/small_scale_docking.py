from pathlib import Path
from shutil import rmtree
from typing import Optional

from asapdiscovery.data.aws.cloudfront import CloudFront
from asapdiscovery.data.aws.s3 import S3
from asapdiscovery.data.dask_utils import (
    DaskType,
    dask_cluster_from_type,
    set_dask_config,
)
from asapdiscovery.data.deduplicator import LigandDeDuplicator
from asapdiscovery.data.fitness import target_has_fitness_data
from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.metadata.resources import master_structures
from asapdiscovery.data.postera.manifold_artifacts import (
    ArtifactType,
    ManifoldArtifactUploader,
)
from asapdiscovery.data.postera.manifold_data_validation import (
    TargetProteinMap,
    map_output_col_to_manifold_tag,
    rename_output_columns_for_manifold,
)
from asapdiscovery.data.postera.molecule_set import MoleculeSetAPI
from asapdiscovery.data.postera.postera_factory import PosteraFactory
from asapdiscovery.data.postera.postera_uploader import PosteraUploader
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.fragalysis import FragalysisFactory
from asapdiscovery.data.schema_v2.ligand import write_ligands_to_multi_sdf
from asapdiscovery.data.schema_v2.molfile import MolFileFactory
from asapdiscovery.data.schema_v2.structure_dir import StructureDirFactory
from asapdiscovery.data.selectors.mcs_selector import MCSSelector
from asapdiscovery.data.services_config import (
    CloudfrontSettings,
    PosteraSettings,
    S3Settings,
)
from asapdiscovery.data.utils import check_empty_dataframe
from asapdiscovery.dataviz.viz_v2.gif_viz import GIFVisualizerV2
from asapdiscovery.dataviz.viz_v2.html_viz import ColourMethod, HTMLVisualizerV2
from asapdiscovery.docking.docking_data_validation import (
    DockingResultColsV2 as DockingResultCols,
)
from asapdiscovery.docking.openeye import POSITDocker
from asapdiscovery.docking.scorer_v2 import ChemGauss4Scorer, MetaScorer, MLModelScorer
from asapdiscovery.docking.workflows.workflows import PosteraDockingWorkflowInputs
from asapdiscovery.ml.models import ASAPMLModelRegistry
from asapdiscovery.modeling.protein_prep_v2 import ProteinPrepper
from asapdiscovery.simulation.simulate import OpenMMPlatform
from asapdiscovery.simulation.simulate_v2 import VanillaMDSimulatorV2
from distributed import Client
from pydantic import Field, PositiveInt, root_validator, validator


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
    ml_scorers : ModelType, optional
        The name of the ml scorers to use.
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
        1, description="Number of targets to dock each ligand against."
    )

    ml_scorers: Optional[list[str]] = Field(
        None, description="The name of the ml scorers to use"
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

    @root_validator
    @classmethod
    def dask_type_cannot_be_lilac_cpu_and_md(cls, values):
        """
        Validate that the dask type is not lilac cpu if MD is requested
        """
        dask_type = values.get("dask_type")
        md = values.get("md")

        if dask_type == dask_type.LILAC_CPU and md:
            raise ValueError("Cannot run MD on a CPU cluster, please use a GPU cluster")
        return values


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
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir()

    logger = FileLogger(
        inputs.logname,  # default root logger so that dask logging is forwarded
        path=output_dir,
        logfile="small-scale-docking.log",
        stdout=True,
        level=inputs.loglevel,
    ).getLogger()

    logger.info(f"Running small scale docking with inputs: {inputs}")
    logger.info(f"Dumping input schema to {output_dir / 'inputs.json'}")

    inputs.to_json_file(output_dir / "small_scale_docking_inputs.json")

    if inputs.use_dask:
        logger.info(f"Using dask for parallelism of type: {inputs.dask_type}")
        set_dask_config()
        dask_cluster = dask_cluster_from_type(inputs.dask_type)
        if inputs.dask_type.is_lilac():
            logger.info("Lilac HPC config selected, setting adaptive scaling")
            dask_cluster.adapt(
                minimum=inputs.dask_cluster_n_workers,
                maximum=inputs.dask_cluster_max_workers,
                wait_count=10,
                interval="1m",
            )
            logger.info(f"Estimating {inputs.dask_cluster_n_workers} workers")
            dask_cluster.scale(inputs.dask_cluster_n_workers)

        dask_client = Client(dask_cluster)
        dask_client.forward_logging()
        logger.info(f"Using dask client: {dask_client}")
        logger.info(f"Using dask cluster: {dask_cluster}")
        logger.info(f"Dask client dashboard: {dask_client.dashboard_link}")

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

    if inputs.postera:
        # can specify postera without uploading
        postera_settings = PosteraSettings()
        logger.info("Postera settings loaded")
        logger.info(
            f"Loading Postera database molecule set {inputs.postera_molset_name}"
        )

        postera = PosteraFactory(
            settings=postera_settings, molecule_set_name=inputs.postera_molset_name
        )
        query_ligands = postera.pull()
    else:
        # load from file
        logger.info(f"Loading ligands from file: {inputs.ligands}")
        molfile = MolFileFactory(filename=inputs.ligands)
        query_ligands = molfile.load()

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
    logger.info("Deduplicating by Inchikey")
    query_ligands = LigandDeDuplicator().deduplicate(query_ligands)
    n_complexes = len(complexes)
    logger.info(f"Loaded {n_complexes} complexes")

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
        ref_chain="A",
        active_site_chain="A",
    )
    prepped_complexes = prepper.prep(
        complexes,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
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
        output_dir=output_dir / "docking_results",
        use_dask=inputs.use_dask,
        dask_client=dask_client,
    )

    n_results = len(results)
    logger.info(f"Docked {n_results} pairs successfully")
    if n_results == 0:
        raise ValueError("No docking results generated, exiting")
    del pairs

    # add chemgauss4 scorer
    scorers = [ChemGauss4Scorer()]

    # load ml scorers
    if inputs.ml_scorers:
        for ml_scorer in inputs.ml_scorers:
            logger.info(f"Loading ml scorer: {ml_scorer}")
            scorers.append(
                MLModelScorer.from_latest_by_target_and_type(inputs.target, ml_scorer)
            )

    if inputs.write_final_sdf:
        logger.info("Writing final docked poses to SDF file")
        write_ligands_to_multi_sdf(
            output_dir / "docking_results.sdf", [r.posed_ligand for r in results]
        )

    # score results
    logger.info("Scoring docking results")
    scorer = MetaScorer(scorers=scorers)
    scores_df = scorer.score(
        results, use_dask=inputs.use_dask, dask_client=dask_client, return_df=True
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
    html_visualizer = HTMLVisualizerV2(
        colour_method=ColourMethod.subpockets,
        target=inputs.target,
        output_dir=html_ouptut_dir,
    )
    pose_visualizatons = html_visualizer.visualize(
        results, use_dask=inputs.use_dask, dask_client=dask_client
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
        html_fitness_visualizer = HTMLVisualizerV2(
            colour_method=ColourMethod.fitness,
            target=inputs.target,
            output_dir=html_fitness_output_dir,
        )
        fitness_visualizations = html_fitness_visualizer.visualize(
            results, use_dask=inputs.use_dask, dask_client=dask_client
        )

        # duplicate target id column so we can join
        fitness_visualizations[
            DockingResultCols.DOCKING_STRUCTURE_POSIT.value
        ] = fitness_visualizations[DockingResultCols.TARGET_ID.value]

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

    if inputs.md:
        if inputs.allow_dask_cuda and inputs.dask_type == DaskType.LOCAL:
            logger.info(
                "Using local CPU dask cluster, and MD has been requested, replacing with a GPU cluster"
            )
            dask_cluster = dask_cluster_from_type(DaskType.LOCAL_GPU)
            dask_client = Client(dask_cluster)
            # dask_client.forward_logging() distributed vs dask_cuda versioning issue, see # #669
            logger.info(f"Using dask client: {dask_client}")
            logger.info(f"Using dask cluster: {dask_cluster}")
            logger.info(f"Dask client dashboard: {dask_client.dashboard_link}")

        md_output_dir = output_dir / "md"

        # capsid simulations need a CA rmsd restraint to hold the capsid together
        if TargetProteinMap[inputs.target] == "Capsid":
            logger.info("Adding CA RMSD restraint to capsid simulation")
            rmsd_restraint = True
            rmsd_restraint_type = "CA"
        else:
            rmsd_restraint = False
            rmsd_restraint_type = None

        md_simulator = VanillaMDSimulatorV2(
            output_dir=md_output_dir,
            openmm_platform=inputs.md_openmm_platform,
            num_steps=inputs.md_steps,
            reporting_interval=inputs.md_report_interval,
            rmsd_restraint=rmsd_restraint,
            rmsd_restraint_type=rmsd_restraint_type,
        )
        simulation_results = md_simulator.simulate(
            results, use_dask=inputs.use_dask, dask_client=dask_client
        )

        gif_output_dir = output_dir / "gifs"
        gif_maker = GIFVisualizerV2(output_dir=gif_output_dir, target=inputs.target)
        gifs = gif_maker.visualize(
            simulation_results, use_dask=inputs.use_dask, dask_client=dask_client
        )

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
        postera_uploader = PosteraUploader(
            settings=PosteraSettings(), molecule_set_name=inputs.postera_molset_name
        )

        # push the results to PostEra, making a new molecule set if necessary
        posit_score_tag = map_output_col_to_manifold_tag(
            DockingResultCols, inputs.target
        )[DockingResultCols.DOCKING_SCORE_POSIT.value]
        result_df, molset_name, made_new_molset = postera_uploader.push(
            result_df, sort_column=posit_score_tag, sort_ascending=True
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
            inputs.target,
            result_df,
            molset_name,
            bucket_name=aws_s3_settings.BUCKET_NAME,
            artifact_types=artifact_types,
            artifact_columns=artifact_columns,
            moleculeset_api=MoleculeSetAPI.from_settings(postera_settings),
            s3=S3.from_settings(aws_s3_settings),
            cloudfront=CloudFront.from_settings(aws_cloudfront_settings),
        )
        uploader.upload_artifacts()
