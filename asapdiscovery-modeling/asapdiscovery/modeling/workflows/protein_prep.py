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
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.fragalysis import FragalysisFactory
from asapdiscovery.data.schema_v2.structure_dir import StructureDirFactory
from asapdiscovery.data.sequence import seqres_by_target
from asapdiscovery.modeling.protein_prep_v2 import CacheType, ProteinPrepper
from distributed import Client
from pydantic import BaseModel, Field, PositiveInt, root_validator, validator


class ProteinPrepInputs(BaseModel):
    """
    Inputs for Protein Prep

    Parameters
    ----------
    target : TargetTags
        The target to prep
    pdb_file : Optional[str]
        Path to a PDB file to prep
    fragalysis_dir : Optional[str]
        Path to a fragalysis dump to prep
    structure_dir : Optional[str]
        Path to a directory of structures to prep
    gen_cache : Path
        Path to a directory to store generated structures
    cache_types : CacheType
        Type of cache to make
    align : Optional[Path]
        Path to a reference structure to align to
    ref_chain : Optional[str]
        Chain ID to align to
    active_site_chain : Optional[str]
        Active site chain ID to align to
    seqres_yaml : Optional[Path]
        Path to a seqres yaml to mutate to
    loop_db : Optional[Path]
        Path to a loop database to use for prepping
    oe_active_site_residue : Optional[str]
        OE formatted string of active site residue to use if not ligand bound
    use_dask : bool
        Whether to use dask for parallelism
    dask_type : DaskType
        Dask client to use for parallelism
    dask_cluster_n_workers : PositiveInt
        Number of workers to use as inital guess for Lilac dask cluster
    dask_cluster_max_workers : PositiveInt
        Maximum number of workers to use for Lilac dask cluster
    logname : str
        Name of the log file
    loglevel : int
        Logging level
    output_dir : Path
        Output directory
    """

    target: TargetTags = Field(None, description="The target to prep")

    pdb_file: Optional[Path] = Field(None, description="Path to a PDB file.")

    fragalysis_dir: Optional[Path] = Field(
        None, description="Path to a directory containing a Fragalysis dump."
    )
    structure_dir: Optional[Path] = Field(
        None,
        description="Path to a directory containing structures to dock instead of a full fragalysis database.",
    )
    gen_cache: str = Field(
        "prepped_structure_cache",
        description="Path to a directory where generated prepped complexes should be cached",
    )

    cache_type: Optional[list[str]] = Field(
        [CacheType.DesignUnit], description="The types of cache to use."
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

    logname: str = Field("protein_prep", description="Name of the log file.")

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

        # can only specify one of fragalysis dir, structure dir and PDB file
        if sum([bool(fragalysis_dir), bool(structure_dir), bool(pdb_file)]) != 1:
            raise ValueError(
                "Must specify exactly one of fragalysis_dir, structure_dir or pdb_file"
            )

        return values

    @validator("cache_type")
    @classmethod
    def check_cache_type(cls, v):
        # must be unique
        if len(v) != len(set(v)):
            raise ValueError("cache_type must be unique")
        return v


def protein_prep_workflow(inputs: ProteinPrepInputs):
    output_dir = inputs.output_dir

    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir()

    logger = FileLogger(
        inputs.logname, path=output_dir, stdout=True, level=inputs.loglevel
    ).getLogger()

    logger.info(f"Running large scale docking with inputs: {inputs}")
    logger.info(f"Dumping input schema to {output_dir / 'inputs.json'}")

    inputs.to_json_file(output_dir / "protein_prep.json")

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

    logger.info(f"Loaded {len(complexes)} complexes")

    if not inputs.seqres_yaml:
        logger.info(
            f"No seqres yaml specified, selecting based on target: {inputs.target}"
        )
        inputs.seqres_yaml = seqres_by_target(inputs.target)

    if inputs.align:
        # load reference structure
        logger.info(f"Loading and aligning to reference structure: {inputs.align}")
        ref_complex = Complex.from_pdb(
            inputs.align,
            target_kwargs={"target_name": "ref"},
            ligand_kwargs={"compound_name": "ref_ligand"},
        )
    else:
        ref_complex = None
    # prep complexes
    logger.info("Prepping complexes")
    prepper = ProteinPrepper(
        loop_db=inputs.loop_db,
        seqres_yaml=inputs.seqres_yaml,
        oe_active_site_residue=inputs.oe_active_site_residue,
        align=ref_complex,
        ref_chain=inputs.ref_chain,
        active_site_chain=inputs.active_site_chain,
    )
    prepped_complexes = prepper.prep(
        inputs=complexes, use_dask=inputs.use_dask, dask_client=dask_client
    )
    logger.info(f"Prepped {len(prepped_complexes)} complexes")
    del complexes

    # cache prepped complexes
    cache_path = output_dir / inputs.gen_cache

    logger.info(f"Caching prepped complexes to {cache_path}")
    for cache_type in inputs.cache_type:
        prepper.cache(prepped_complexes, cache_path, type=cache_type)

    logger.info("Done")