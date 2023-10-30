from pydantic import BaseModel, Field, root_validator, PositiveInt

from enum import Enum
from pathlib import Path
from typing import Optional
import logging
from shutil import rmtree


class ProteinPrepInputs(BaseModel):
    target: TargetTags = Field(None, description="The target to dock against.")

    pdb_file: Optional[str] = Field(None, description="Path to a PDB file.")

    fragalysis_dir: Optional[str] = Field(
        None, description="Path to a directory containing a Fragalysis dump."
    )
    structure_dir: Optional[str] = Field(
        None,
        description="Path to a directory containing structures to dock instead of a full fragalysis database.",
    )
    gen_cache: Path = Field(
        Path("gen_cache"),
        description="Path to a directory where generated prepped complexes should be cached",
    )

    cache_types: CacheType = Field(
        CacheType.DesignUnit, description="Type of cache to make"
    )

    align: Optional[Path] = Field(
        None, description="Reference structure pdb to align to."
    )
    ref_chain: Optional[str] = Field(
        None, description="Reference chain ID to align to."
    )
    active_site_chain: Optional[str] = Field(
        None, description="Chain ID to align to reference."
    )
    seqres_yaml: Optional[Path] = Field(
        None, description="Path to seqres yaml to mutate to."
    )
    loop_db: Optional[Path] = Field(
        None, description="Path to loop database to use for prepping"
    )
    oe_active_site_residue: Optional[str] = Field(
        None, description="OE formatted string of active site residue to use"
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
        40, description="Maximum number of workers to use for Lilac dask cluster"
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
        fragalysis_dir = values.get("fragalysis_dir")
        structure_dir = values.get("structure_dir")
        pdb_file = values.get("pdb_file")

        if pdb_file and fragalysis_dir:
            raise ValueError("Cannot specify both pdb_file and fragalysis_dir.")

        if pdb_file and structure_dir:
            raise ValueError("Cannot specify both pdb_file and structure_dir.")

        if fragalysis_dir and structure_dir:
            raise ValueError("Cannot specify both fragalysis_dir and structure_dir.")

        if not fragalysis_dir and not structure_dir:
            raise ValueError("Must specify either fragalysis_dir or structure_dir.")

        return values


def protein_prep(inputs: ProteinPrepInputs):
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
                minimum=1,
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
        complex = Complex.from_pdb(inputs.pdb_file)
        complexes = [complex]

    else:
        raise ValueError(
            "Must specify either fragalysis_dir, structure_dir or pdb_file"
        )

    logger.info(f"Loaded {len(complexes)} complexes")

    if not inputs.seqres_yaml:
        logger.info(
            f"No seqres yaml specified, selecting based on target: {inputs.target}"
        )
        inputs.seqres_yaml = select_seqres_yaml(inputs.target)

    # prep complexes
    logger.info("Prepping complexes")
    prepper = ProteinPrepper(
        loop_db=inputs.loop_db,
        seqres_yaml=inputs.seqres_yaml,
        oe_active_site_residue=inputs.oe_active_site_residue,
        align=inputs.align,
        ref_chain=inputs.ref_chain,
        active_site_chain=inputs.active_site_chain,
    )
    prepped_complexes = prepper.prep(
        inputs=complexes, use_dask=inputs.use_dask, dask_client=dask_client
    )
    logger.info(f"Prepped {len(prepped_complexes)} complexes")

    # cache prepped complexes
    logger.info("Caching prepped complexes")
    prepper.cache(prepped_complexes, inputs.gen_du_cache, type=CacheType.DesignUnit)

    logger.info("Done")
