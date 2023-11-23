"""
A test-oriented docking workflow for testing the docking pipeline.
Removes all the additional layers in the other workflows and adds some features to make running cross-docking easier
"""
from pathlib import Path
from shutil import rmtree

from asapdiscovery.data.dask_utils import dask_cluster_from_type, set_dask_config
from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.postera.manifold_data_validation import (
    rename_output_columns_for_manifold,
)
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.fragalysis import FragalysisFactory
from asapdiscovery.data.schema_v2.ligand import write_ligands_to_multi_sdf
from asapdiscovery.data.schema_v2.molfile import MolFileFactory
from asapdiscovery.data.schema_v2.structure_dir import StructureDirFactory
from asapdiscovery.data.selectors.selector_list import StructureSelector
from asapdiscovery.docking.docking_data_validation import (
    DockingResultColsV2 as DockingResultCols,
)
from asapdiscovery.docking.docking_v2 import DockingInputMultiStructure
from asapdiscovery.docking.openeye import POSIT_METHOD, POSIT_RELAX_MODE, POSITDocker
from asapdiscovery.docking.scorer_v2 import ChemGauss4Scorer, MetaScorer
from asapdiscovery.docking.workflows.workflows import DockingWorkflowInputsBase
from asapdiscovery.modeling.protein_prep_v2 import ProteinPrepper
from distributed import Client
from pydantic import Field, PositiveInt


class CrossDockingWorkflowInputs(DockingWorkflowInputsBase):
    logname: str = Field("cross_docking", description="Name of the log file.")

    structure_selector: StructureSelector = Field(
        StructureSelector.PAIRWISE,
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
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir()

    logger = FileLogger(
        inputs.logname, path=output_dir, stdout=True, level=inputs.loglevel
    ).getLogger()

    logger.info(f"Running cross docking with inputs: {inputs}")
    logger.info(f"Dumping input schema to {output_dir / 'inputs.json'}")

    inputs.to_json_file(output_dir / "cross_docking_inputs.json")

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
    logger.info(f"Loading ligands from file: {inputs.ligands}")
    molfile = MolFileFactory.from_file(inputs.ligands)
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
    logger.info("Selecting pairs for docking")
    # TODO: MCS takes an n_select arg but Pairwise does not...meaning we are losing that option the way this is written
    selector = inputs.structure_selector.selector_cls()
    pairs = selector.select(
        query_ligands,
        prepped_complexes,
        use_dask=False,
        dask_client=None,
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
