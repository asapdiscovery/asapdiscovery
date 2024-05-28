from pathlib import Path
from shutil import rmtree
from typing import Optional

from asapdiscovery.data.metadata.resources import master_structures
from asapdiscovery.data.operators.deduplicator import LigandDeDuplicator
from asapdiscovery.data.operators.selectors.mcs_selector import MCSSelector
from asapdiscovery.data.readers.meta_ligand_factory import MetaLigandFactory
from asapdiscovery.data.readers.meta_structure_factory import MetaStructureFactory
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.schema.ligand import write_ligands_to_multi_sdf
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
from asapdiscovery.data.util.dask_utils import DaskType, make_dask_client_meta
from asapdiscovery.data.util.logging import FileLogger
from asapdiscovery.data.util.utils import check_empty_dataframe
from asapdiscovery.dataviz.gif_viz import GIFVisualizer
from asapdiscovery.dataviz.html_viz import ColorMethod, HTMLVisualizer
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
from asapdiscovery.simulation.simulate import OpenMMPlatform, VanillaMDSimulator
from asapdiscovery.workflows.docking_workflows.workflows import (
    PosteraDockingWorkflowInputs,
)


class SymExpCrystalPackingInputs(PosteraDockingWorkflowInputs):
    ...
   

def symexp_crystal_packing_workflow(inputs: SymExpCrystalPackingInputs):
    """
    Run large scale docking on a set of ligands, against multiple targets

    Parameters
    ----------
    inputs : SymExpCrystalPackingInputs
        Inputs to symexp crystal packing workflow

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
        logfile="symexp-crystal-packing.log",
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
    inputs.to_json_file(output_dir / "symexp_crystal_packing_inputs.json")

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
    selector = MCSSelector()
    pairs = selector.select(
        query_ligands,
        prepped_complexes,
        n_select=inputs.n_select,
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
        failure_mode=inputs.failure_mode,
    )

    n_results = len(results)
    logger.info(f"Docked {n_results} pairs successfully")
    if n_results == 0:
        raise ValueError("No docking results generated, exiting")
    del pairs

    # score results with multiple scoring functions
    logger.info("Scoring docking results")
    scorer = MetaScorer(scorers=scorers)
    scores_df = scorer.score(
        results,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        failure_mode=inputs.failure_mode,
        return_df=True,
        include_input=True,
    )

    scores_df.to_csv(data_intermediates / "docking_scores_raw.csv", index=False)


    logger.info("Symmetry expanding docked structures")
    expander = SymmetryExpander()

    expanded_complexes = expander.expand(
        results,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        failure_mode=inputs.failure_mode,
    )

    logger.info("Writing expanded structures to PDB")

    expanded_pdb_dir = output_dir / "expanded_pdbs"
    expanded_pdb_dir.mkdir(exist_ok=True)

    [c.to_pdb_file(expanded_pdb_dir / f"{c.hash}.pdb") for c in expanded_complexes]

    # score results using multiple scoring functions
    logger.info("Scoring expanded structures")
   
    scorer = SymExpClashScorer()

    logger.info("Running scoring")
    scores_df = scorer.score(
        expanded_complexes,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        failure_mode=inputs.failure_mode,
        return_df=True,
    )

    # set hit flag to False
    scores_df[DockingResultCols.SYMEXP_CLASHING.value] = False

   # if clashing is greater than threshold, set hit flag to True
    scores_df.loc[
        scores_df[DockingResultCols.SYMEXP_CLASHING.value] > inputs.clash_threshold,
        DockingResultCols.SYMEXP_CLASHING.value,
    ] = True

    clashing_df = scores_df[  # noqa: E712
        scores_df[DockingResultCols.SYMEXP_CLASHING.value] == True  # noqa: E712
    ]

    non_clashing_df = scores_df[
        scores_df[DockingResultCols.SYMEXP_CLASHING.value] == False
    ]

    # write to csv
    clashing_df.to_csv(output_dir / "clashing.csv", index=False)
    non_clashing_df.to_csv(output_dir / "non_clashing.csv", index=False)
    scores_df.to_csv(output_dir / "raw_scores.csv", index=False)

    # run html visualiser to get web-ready vis of docked poses in expanded form
    logger.info("Running HTML visualiser for poses")
    html_ouptut_dir = output_dir / "poses"
    html_visualizer = HTMLVisualizer(
        color_method=ColorMethod.subpockets,
        target=inputs.target,
        output_dir=html_ouptut_dir,
    )
    pose_visualizatons = html_visualizer.visualize(
        expanded_complexes,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        failure_mode=inputs.failure_mode,
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
   
    # rename columns for manifold
    logger.info("Renaming columns for manifold")


    # deduplicate scores_df TODO


    # rename columns for manifold
    result_df = rename_output_columns_for_manifold(
        scores_df,
        inputs.target,
        [DockingResultCols],
        manifold_validate=True,
        drop_non_output=True,
        allow=[
            DockingResultCols.LIGAND_ID.value,
            DockingResultCols.HTML_PATH_POSE.value,
        ],
    )

    result_df.to_csv(output_dir / "symexp_final.csv", index=False)

    if inputs.postera_upload:
        logger.info("Uploading results to Postera")
        posit_score_tag = map_output_col_to_manifold_tag(
            DockingResultCols, inputs.target
        )[DockingResultCols.DOCKING_SCORE_POSIT.value]

        postera_uploader = PosteraUploader(
            settings=PosteraSettings(),
            molecule_set_name=inputs.postera_molset_name,
        )
        # push the results to PostEra, making a new molecule set if necessary
        manifold_data, molset_name, made_new_molset = postera_uploader.push(
            result_df
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
