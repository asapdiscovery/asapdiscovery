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
from typing import Optional, Union

from asapdiscovery.data.operators.selectors.selector_list import StructureSelector
from asapdiscovery.data.readers.meta_structure_factory import MetaStructureFactory
from asapdiscovery.data.readers.molfile import MolFileFactory
from asapdiscovery.data.schema.ligand import write_ligands_to_multi_sdf
from asapdiscovery.data.services.postera.manifold_data_validation import (
    TargetProteinMap,
    rename_output_columns_for_manifold,
)
from asapdiscovery.data.util.dask_utils import DaskType, make_dask_client_meta
from asapdiscovery.data.util.logging import FileLogger
from asapdiscovery.docking.docking import DockingInputMultiStructure
from asapdiscovery.docking.docking_data_validation import DockingResultCols
from asapdiscovery.docking.openeye import POSIT_METHOD, POSIT_RELAX_MODE, POSITDocker
from asapdiscovery.docking.scorer import (
    ChemGauss4Scorer,
    FINTScorer,
    MetaScorer,
    MLModelScorer,
)
from asapdiscovery.modeling.protein_prep import LigandTransferProteinPrepper
from asapdiscovery.dataviz.gif_viz import GIFVisualizer
from asapdiscovery.dataviz.html_viz import ColorMethod, HTMLVisualizer
# from asapdiscovery.workflows.docking_workflows.workflows import WorkflowInputsBase
from asapdiscovery.workflows.docking_workflows.workflows import DockingWorkflowInputsBase
from pydantic import Field, PositiveInt, validator, root_validator

from asapdiscovery.simulation.simulate import OpenMMPlatform, VanillaMDSimulator

class LigandTransferDockingWorkflowInputs(DockingWorkflowInputsBase):#WorkflowInputsBase):
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

    # copied form SmallScaleDockingInputs

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
            loglevel=inputs.loglevel,
            n_workers=inputs.dask_n_workers,
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
        failure_mode=inputs.failure_mode,
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
        failure_mode=inputs.failure_mode,
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
    logger.info("Prepping complexes")
    prepped_complexes = prepper.prep(
        targets,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        failure_mode=inputs.failure_mode,
        cache_dir=inputs.cache_dir,
        use_only_cache=inputs.use_only_cache,
    )
    del targets

    n_prepped_complexes = len(prepped_complexes)
    logger.info(f"Prepped {n_prepped_complexes} complexes")
    for pc in prepped_complexes:
        logger.info(f"Complex is {pc.target.target_name}")

    if inputs.save_to_cache and inputs.cache_dir is not None:
        logger.info(f"Writing prepped complexes to global cache {inputs.cache_dir}")
        prepper.cache(prepped_complexes, inputs.cache_dir)

    # Here the only thing that makes sense is the self docking selector
    selector = StructureSelector.SELF_DOCKING.selector_cls()
    ligands = [pc.ligand for pc in prepped_complexes]
    pairs = selector.select(ligands, prepped_complexes)

    n_pairs = len(pairs)
    logger.info(
        f"Selected {n_pairs} pairs for docking, from {len(ligands)} ligands and {n_prepped_complexes} compexes"
    )
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
        failure_mode=inputs.failure_mode,
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

    # score results
    logger.info("Scoring docking results")
    scorer = MetaScorer(scorers=scorers)

    if inputs.write_final_sdf:
        logger.info("Writing final docked poses to SDF file")
        write_ligands_to_multi_sdf(
            output_dir / "docking_results.sdf", [r.posed_ligand for r in results]
        )
    # NOTE: The return_df option doesn't work, the DataFrame.pivot() function fails. Fix is pending.
    scores_list = scorer.score(
        results,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        return_df=False,
        failure_mode=inputs.failure_mode,
    )
    import csv
    with open(data_intermediates / "docking_scores_raw.csv", "w", newline="") as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for item in scores_list:
            wr.writerow([item])

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
    failure_mode=inputs.failure_mode,
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
        )    

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
