import logging
from pathlib import Path
from shutil import rmtree
from typing import Optional

from asapdiscovery.data.dask_utils import (
    DaskType,
    dask_client_from_type,
    set_dask_config,
)
from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.postera.manifold_data_validation import (
    TargetTags,
    rename_output_columns_for_manifold,
)
from asapdiscovery.data.postera.postera_factory import PosteraFactory
from asapdiscovery.data.postera.postera_uploader import PosteraUploader
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
from asapdiscovery.docking.docking_v2 import POSITDocker
from asapdiscovery.docking.scorer_v2 import (
    ChemGauss4Scorer,
    GATScorer,
    MetaScorer,
    SchnetScorer,
)
from asapdiscovery.modeling.protein_prep_v2 import ProteinPrepper
from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    PositiveInt,
    root_validator,
    validator,
)


class LargeScaleDockingInputs(BaseModel):
    filename: Optional[str] = Field(
        None, description="Path to a molecule file containing query ligands."
    )
    fragalysis_dir: Optional[str] = Field(
        None, description="Path to a directory containing a Fragalysis dump."
    )
    structure_dir: Optional[str] = Field(
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
    du_cache: Optional[str] = Field(
        None, description="Path to a directory where design units are cached"
    )

    gen_du_cache: Optional[str] = Field(
        None,
        description="Path to a directory where generated design units should be cached",
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

    n_select: PositiveInt = Field(
        10, description="Number of targets to dock each ligand against, sorted by MCS"
    )

    top_n: PositiveInt = Field(
        500, description="Number of docking results to return, ordered by docking score"
    )

    posit_confidence_cutoff: PositiveFloat = Field(
        0.7, description="POSIT confidence cutoff used to filter docking results"
    )

    logname: str = Field("large_scale_docking.log", description="Name of the log file.")

    loglevel: int = Field(logging.INFO, description="Logging level.")

    output_dir: Path = Field(Path("output"), description="Output directory")

    class Config:
        arbitrary_types_allowed = True

    @validator("posit_confidence_cutoff")
    @classmethod
    def posit_confidence_cutoff_must_be_between_0_and_1(cls, v):
        """
        Validate that the POSIT confidence cutoff is between 0 and 1
        """
        if v < 0 or v > 1:
            raise ValueError("POSIT confidence cutoff must be between 0 and 1.")
        return v

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
        du_cache = values.get("du_cache")
        gen_du_cache = values.get("gen_du_cache")

        if postera and filename:
            raise ValueError("Cannot specify both filename and postera.")

        if not postera and not filename:
            raise ValueError("Must specify either filename or postera.")

        if postera_upload and not postera_molset_name:
            raise ValueError(
                "Must specify postera_molset_name if uploading to postera."
            )

        if fragalysis_dir and structure_dir:
            raise ValueError("Cannot specify both fragalysis_dir and structure_dir.")

        if not fragalysis_dir and not structure_dir:
            raise ValueError("Must specify either fragalysis_dir or structure_dir.")

        if du_cache and gen_du_cache:
            raise ValueError("Cannot specify both du_cache and gen_du_cache.")

        return values

    @validator("du_cache")
    @classmethod
    def du_cache_must_be_directory(cls, v):
        """
        Validate that the DU cache is a directory
        """
        if v is not None:
            if not Path(v).is_dir():
                raise ValueError("Du cache must be a directory.")
        return v


def large_scale_docking(inputs: LargeScaleDockingInputs):
    """
    Run large scale docking on a set of ligands, against multiple targets
    """

    output_dir = inputs.output_dir
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir()

    logger = FileLogger(
        inputs.logname, path=output_dir, stdout=True, level=inputs.loglevel
    ).getLogger()

    logger.info(f"Running large scale docking with inputs: {inputs}")

    if inputs.use_dask:
        set_dask_config()
        logger.info("Using dask for parallelism.")
        dask_client = dask_client_from_type(inputs.dask_type)
        logger.info(f"Using dask client: {dask_client}")
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

    # load complexes from a directory or from fragalysis
    if inputs.structure_dir:
        logger.info(f"Loading structures from directory: {inputs.structure_dir}")
        structure_factory = StructureDirFactory.from_dir(inputs.structure_dir)
        complexes = structure_factory.load(
            use_dask=inputs.use_dask, dask_client=dask_client
        )
    else:
        logger.info(f"Loading structures from fragalysis: {inputs.fragalysis_dir}")
        fragalysis = FragalysisFactory.from_dir(inputs.fragalysis_dir)
        complexes = fragalysis.load(use_dask=inputs.use_dask, dask_client=dask_client)

    # prep complexes
    logger.info("Prepping complexes")
    prepper = ProteinPrepper(du_cache=inputs.du_cache)
    prepped_complexes = prepper.prep(
        complexes, use_dask=inputs.use_dask, dask_client=dask_client
    )
    del complexes

    if inputs.gen_du_cache:
        logger.info(f"Generating DU cache at {inputs.gen_du_cache}")
        prepper.cache(prepped_complexes, inputs.gen_du_cache)

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

    del prepped_complexes

    # dock pairs
    logger.info("Running docking on selected pairs")
    docker = POSITDocker()
    results = docker.dock(
        pairs,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
    )

    del pairs

    # write docking results
    logger.info("Writing docking results")
    POSITDocker.write_docking_files(results, output_dir / "docking_results")

    # score results
    logger.info("Scoring docking results")
    scorer = MetaScorer(
        scorers=[
            ChemGauss4Scorer(),
            GATScorer.from_latest_by_target(inputs.target),
            SchnetScorer.from_latest_by_target(inputs.target),
        ]
    )

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

    check_empty_dataframe(
        scores_df,
        logger=logger,
        fail="raise",
        tag="scores",
        message="No docking results passed the POSIT confidence cutoff",
    )

    # filter out clashes (chemgauss4 score > 0)
    scores_df = scores_df[scores_df[DockingResultCols.DOCKING_SCORE_POSIT] <= 0]

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

    # take top n results
    scores_df = scores_df.head(inputs.top_n)

    check_empty_dataframe(
        scores_df,
        logger=logger,
        fail="raise",
        tag="scores",
        message="No docking results passed the top n filter",
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
