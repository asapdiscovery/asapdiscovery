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
from asapdiscovery.data.selectors.selector_list import StructureSelector
from asapdiscovery.data.selectors.mcs_selector import MCSSelector
from asapdiscovery.data.services_config import PosteraSettings
from asapdiscovery.data.utils import check_empty_dataframe
from asapdiscovery.docking.docking_data_validation import (
    DockingResultColsV2 as DockingResultCols,
)
from asapdiscovery.docking.docking_v2 import POSITDocker
from asapdiscovery.docking.docking_methods import DockingMethod
from asapdiscovery.docking.scorer_v2 import ChemGauss4Scorer, MetaScorer, MLModelScorer
from asapdiscovery.ml.models.ml_models import ASAPMLModelRegistry
from asapdiscovery.modeling.protein_prep_v2 import CacheType, ProteinPrepper
from distributed import Client
from pydantic import BaseModel, Field, PositiveInt, root_validator, validator


class LargeScaleDockingInputs(BaseModel):
    """
    Schema for inputs to large scale docking

    Parameters
    ----------
    filename : str, optional
        Path to a molecule file containing query ligands.
    fragalysis_dir : str, optional
        Path to a directory containing a Fragalysis dump.
    structure_dir : str, optional
        Path to a directory containing structures to dock instead of a full fragalysis database.
    postera : bool, optional
        Whether to use the Postera database as the query set.
    postera_upload : bool, optional
        Whether to upload the results to Postera.
    postera_molset_name : str, optional
        The name of the molecule set to pull from and/or upload to.
    cache_dir : str, optional
        Path to a directory where structures are cached
    gen_cache : str, optional
        Path to a directory where prepped structures should be cached
    cache_type : list[CacheType], optional
        The types of cache to use.
    target : TargetTags, optional
        The target to dock against.
    write_final_sdf : bool, optional
        Whether to write the final docked poses to an SDF file.
    use_dask : bool, optional
        Whether to use dask for parallelism.
    dask_type : DaskType, optional
        Type of dask client to use for parallelism.
    n_select : int, optional
        Number of targets to dock each ligand against, sorted by MCS
    top_n : int, optional
        Number of docking results to return, ordered by docking score
    posit_confidence_cutoff : float, optional
        POSIT confidence cutoff used to filter docking results
    ml_scorers : MLModelType, optional
        The name of the ml scorers to use.
    logname : str, optional
        Name of the log file.
    loglevel : int, optional
        Logging level.
    output_dir : Path, optional
        Output directory
    """

    filename: Optional[str] = Field(
        None, description="Path to a molecule file containing query ligands."
    )

    pdb_file: Optional[Path] = Field(
        None, description="Path to a PDB file to prep and dock to."
    )

    fragalysis_dir: Optional[Path] = Field(
        None, description="Path to a directory containing a Fragalysis dump."
    )
    structure_dir: Optional[Path] = Field(
        None,
        description="Path to a directory containing structures to dock instead of a full fragalysis database.",
    )
    postera: bool = Field(
        False, description="Whether to use the Postera database as the query set."
    )
    postera_upload: bool = Field(
        False, description="Whether to upload the results to Postera."
    )
    postera_molset_name: Optional[str] = Field(
        None, description="The name of the molecule set to upload to."
    )
    cache_dir: Optional[str] = Field(
        None, description="Path to a directory where a cache has been generated"
    )

    gen_cache: Optional[str] = Field(
        None,
        description="Generate a cache from structures prepped in this workflow run in this directory",
    )

    cache_type: Optional[list[str]] = Field(
        [CacheType.DesignUnit], description="The types of cache to use."
    )

    target: TargetTags = Field(None, description="The target to dock against.")

    write_final_sdf: bool = Field(
        default=True,
        description="Whether to write the final docked poses to an SDF file.",
    )
    use_dask: bool = Field(True, description="Whether to use dask for parallelism.")

    dask_type: DaskType = Field(
        DaskType.LOCAL, description="Dask client to use for parallelism."
    )

    dask_cluster_n_workers: PositiveInt = Field(
        10,
        description="Number of workers to use as inital guess for Lilac dask cluster",
    )

    dask_cluster_max_workers: PositiveInt = Field(
        200, description="Maximum number of workers to use for Lilac dask cluster"
    )

    structure_selector: StructureSelector = Field(
        "MCS", description="Structure selector to use for docking"
    )

    n_select: PositiveInt = Field(
        5, description="Number of targets to dock each ligand against, sorted by MCS"
    )

    multi_reference: bool = Field(
        False,
        description="Whether to use multi reference docking, in which the docking_method "
        "recieves a DockingInputMultiStructure object instead of a DockingInputPair object",
    )

    docking_method: DockingMethod = Field("POSIT", description="Docking method to use")

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

    logname: str = Field("large_scale_docking", description="Name of the log file.")

    loglevel: int = Field(logging.INFO, description="Logging level")

    output_dir: Path = Field(Path("output"), description="Output directory")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_json_file(cls, file: str | Path):
        return cls.parse_file(str(file))

    def to_json_file(self, file: str | Path):
        with open(file, "w") as f:
            f.write(self.json(indent=2))

    @root_validator
    @classmethod
    def check_inputs(cls, values):
        """
        Validate inputs
        """
        filename = values.get("filename")
        fragalysis_dir = values.get("fragalysis_dir")
        structure_dir = values.get("structure_dir")
        postera = values.get("postera")
        postera_upload = values.get("postera_upload")
        postera_molset_name = values.get("postera_molset_name")
        cache_dir = values.get("cache_dir")
        gen_cache = values.get("gen_cache")
        pdb_file = values.get("pdb_file")

        if postera and filename:
            raise ValueError("Cannot specify both filename and postera.")

        if not postera and not filename:
            raise ValueError("Must specify either filename or postera.")

        if postera_upload and not postera_molset_name:
            raise ValueError(
                "Must specify postera_molset_name if uploading to postera."
            )

        # can only specify one of fragalysis dir, structure dir and PDB file
        if sum([bool(fragalysis_dir), bool(structure_dir), bool(pdb_file)]) != 1:
            raise ValueError(
                "Must specify exactly one of fragalysis_dir, structure_dir or pdb_file"
            )

        if cache_dir and gen_cache:
            raise ValueError("Cannot specify both cache_dir and gen_cache.")

        return values

    @validator("cache_dir")
    @classmethod
    def cache_dir_must_be_directory(cls, v):
        """
        Validate that the DU cache is a directory
        """
        if v is not None:
            if not Path(v).is_dir():
                raise ValueError("Du cache must be a directory.")
        return v

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
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir()

    logger = FileLogger(
        inputs.logname, path=output_dir, stdout=True, level=inputs.loglevel
    ).getLogger()

    logger.info(f"Running large scale docking with inputs: {inputs}")
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

    if inputs.postera:
        # load postera
        logger.info(
            f"Loading Postera database molecule set {inputs.postera_molset_name}"
        )
        postera_settings = PosteraSettings()
        postera = PosteraFactory(
            settings=postera_settings, molecule_set_name=inputs.postera_molset_name
        )
        query_ligands = postera.pull()
    else:
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
    docker = POSITDocker(
        use_omega=inputs.use_omega, allow_retries=inputs.allow_posit_retries
    )
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

    # load ml scorers
    if inputs.ml_scorers:
        for ml_scorer in inputs.ml_scorers:
            logger.info(f"Loading ml scorer: {ml_scorer}")
            scorer = MLModelScorer.from_latest_by_target_and_type(
                inputs.target, ml_scorer
            )
            if scorer:
                scorers.append(scorer)

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

    if inputs.postera_upload:
        logger.info("Uploading results to Postera")
        postera_uploader = PosteraUploader(
            settings=PosteraSettings(), molecule_set_name=inputs.postera_molset_name
        )
        postera_uploader.push(result_df)
