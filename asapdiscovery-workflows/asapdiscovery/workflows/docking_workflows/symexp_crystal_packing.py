from pathlib import Path
from shutil import rmtree

from asapdiscovery.data.operators.deduplicator import LigandDeDuplicator
from asapdiscovery.data.operators.selectors.mcs_selector import RascalMCESSelector
from asapdiscovery.data.operators.symmetry_expander import SymmetryExpander
from asapdiscovery.data.readers.meta_ligand_factory import MetaLigandFactory
from asapdiscovery.data.readers.meta_structure_factory import MetaStructureFactory
from asapdiscovery.data.schema.ligand import write_ligands_to_multi_sdf
from asapdiscovery.data.services.aws.cloudfront import CloudFront
from asapdiscovery.data.services.aws.s3 import S3
from asapdiscovery.data.services.postera.manifold_artifacts import (
    ArtifactType,
    ManifoldArtifactUploader,
)
from asapdiscovery.data.services.postera.manifold_data_validation import (
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
from asapdiscovery.data.util.dask_utils import make_dask_client_meta
from asapdiscovery.data.util.logging import FileLogger
from asapdiscovery.data.util.utils import check_empty_dataframe
from asapdiscovery.dataviz.html_viz import ColorMethod, HTMLVisualizer
from asapdiscovery.docking.docking_data_validation import DockingResultCols
from asapdiscovery.docking.openeye import POSITDocker
from asapdiscovery.docking.scorer import ChemGauss4Scorer, SymClashScorer
from asapdiscovery.modeling.protein_prep import ProteinPrepper
from asapdiscovery.workflows.docking_workflows.workflows import (
    PosteraDockingWorkflowInputs,
)
from pydantic import Field


class SymExpCrystalPackingInputs(PosteraDockingWorkflowInputs):
    vdw_radii_fudgefactor: float = Field(0.9, description="Fudge factor for VDW radii")
    symexp_clash_thresh: int = Field(
        0,
        description="Clash threshold for symmetry expansion to be considered clashing ( > thresh is clashing)",
    )


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

    # prep complexes
    logger.info("Prepping complexes")
    prepper = ProteinPrepper(
        cache_dir=inputs.cache_dir,
        align=None,
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
    selector = RascalMCESSelector()
    pairs = selector.select(
        query_ligands,
        prepped_complexes,
        n_select=1,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        failure_mode=inputs.failure_mode,
    )

    n_pairs = len(pairs)
    logger.info(f"Selected {n_pairs} pairs for docking")

    logger.info("Bounding boxes and space groups for selected pairs")
    logger.info("Ligand, Target, Box")
    for pair in pairs:
        logger.info(
            f"{pair.ligand.compound_name} {pair.complex.target.target_name} {pair.complex.target.crystal_symmetry}"
        )

    del prepped_complexes

    # dock pairs
    logger.info("Running docking on selected pairs")
    docker = POSITDocker(use_omega=True, allow_retries=True)
    results = docker.dock(
        pairs,
        output_dir=output_dir / "docking_results",
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        failure_mode="skip",
    )

    logger.info("Docking complete")

    complexes = [result.to_posed_complex() for result in results]

    n_results = len(results)
    logger.info(f"Docked {n_results} pairs successfully")
    if n_results == 0:
        raise ValueError("No docking results generated, exiting")
    del pairs

    if inputs.write_final_sdf:
        logger.info("Writing final docked poses to SDF file")
        write_ligands_to_multi_sdf(
            output_dir / "docking_results.sdf", [r.posed_ligand for r in results]
        )

    # score results with just normal ChemGauss scorer
    logger.info("Scoring docking results")
    scorer = ChemGauss4Scorer()
    scores_df = scorer.score(
        results,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        failure_mode=inputs.failure_mode,
        return_df=True,
    )

    scores_df.to_csv(data_intermediates / "docking_scores_raw.csv", index=False)

    check_empty_dataframe(
        scores_df,
        logger=logger,
        fail="raise",
        tag="scores",
        message="No docking results",
    )

    logger.info("Symmetry expanding docked structures")
    expander = SymmetryExpander()
    expanded_complexes = expander.expand(
        complexes,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        failure_mode="raise",
    )

    logger.info("Writing expanded structures to PDB")

    expanded_pdb_dir = output_dir / "expanded_pdbs"
    expanded_pdb_dir.mkdir(exist_ok=True)

    # the PDBs are too big to hash with inchi so use name (not ideal but will work for now)
    [
        c.to_pdb(
            expanded_pdb_dir
            / f"expanded_{c.target.target_name}_{c.ligand.compound_name}.pdb"
        )
        for c in expanded_complexes
    ]

    logger.info("Scoring expanded structures")

    scorer = SymClashScorer()

    logger.info("Running scoring")
    scores_df_exp = scorer.score(
        expanded_complexes,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        failure_mode=inputs.failure_mode,
        return_df=True,
    )

    scores_df_exp.to_csv(data_intermediates / "symexp_scores_raw.csv", index=False)

    # set hit flag to False
    scores_df_exp[DockingResultCols.SYMEXP_CLASHING.value] = False

    # if clashing is greater than threshold, set hit flag to True
    scores_df_exp.loc[
        scores_df_exp[DockingResultCols.SYMEXP_CLASH_NUM.value]
        > inputs.symexp_clash_thresh,
        DockingResultCols.SYMEXP_CLASHING.value,
    ] = True

    # join with original scores
    scores_df = scores_df.merge(
        scores_df_exp,
        on=[
            DockingResultCols.LIGAND_ID.value,
            DockingResultCols.DOCKING_STRUCTURE_POSIT.value,
            DockingResultCols.SMILES.value,
        ],
        how="outer",
    )

    scores_df.to_csv(data_intermediates / "symexp_scores_combined.csv", index=False)

    # # split into clashing and non-clashing
    clashing_df = scores_df[
        scores_df[DockingResultCols.SYMEXP_CLASHING.value] == True  # noqa: E712
    ]

    non_clashing_df = scores_df[
        scores_df[DockingResultCols.SYMEXP_CLASHING.value] == False  # noqa: E712
    ]

    # write to csv
    clashing_df.to_csv(data_intermediates / "clashing.csv", index=False)
    non_clashing_df.to_csv(data_intermediates / "non_clashing.csv", index=False)

    # # run html visualiser to get web-ready vis of docked poses in expanded form
    # logger.info("Running HTML visualiser for poses")
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

    # # deduplicate scores_df TODO

    # rename columns for manifold
    result_df = rename_output_columns_for_manifold(
        combined_df,
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
        uploader.upload_artifacts(sort_column=posit_score_tag, sort_ascending=True)
