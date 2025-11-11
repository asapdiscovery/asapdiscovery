import logging
from pathlib import Path
from typing import Optional, Union

from asapdiscovery.data.metadata.resources import master_structures
from asapdiscovery.data.readers.structure_dir import StructureDirFactory
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.services.fragalysis.fragalysis_reader import FragalysisFactory
from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags
from asapdiscovery.data.util.dask_utils import DaskType, make_dask_client_meta
from asapdiscovery.data.util.logging import FileLogger
from asapdiscovery.modeling.protein_prep import ProteinPrepper
from pydantic.v1 import BaseModel, Field, PositiveInt, root_validator


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
    cache_dir : Path
        Path to a directory of cached prepped structures.
    save_to_cache: bool
        If newly prepared structures should also be saved to the global cache
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
    cache_dir: Optional[str] = Field(
        "prepped_structure_cache",
        description="Path to a directory of cached prepared Complex structures.",
    )
    save_to_cache: bool = Field(
        True,
        description="If newly prepared structures should also be saved to the cache_dir, has no effect if the cache_dir is not set.",
    )

    align: Optional[Path] = Field(
        None, description="Reference structure pdb to align to."
    )
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
    oe_active_site_residue: Optional[str] = Field(
        None, description="OE formatted string of active site residue to use"
    )

    use_dask: bool = Field(True, description="Whether to use dask for parallelism.")

    dask_type: DaskType = Field(
        DaskType.LOCAL, description="Dask client to use for parallelism."
    )

    dask_n_workers: Optional[PositiveInt] = Field(None, description="Number of workers")

    logname: str = Field("", description="Name of the log file.")

    loglevel: Union[str, int] = Field(logging.INFO, description="Logging level")

    output_dir: Path = Field(
        Path("output"),
        description="Output directory where newly prepped structures and log files will be saved to.",
    )

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


def protein_prep_workflow(inputs: ProteinPrepInputs):
    output_dir = inputs.output_dir
    output_dir.mkdir(exist_ok=True)

    logger = FileLogger(
        inputs.logname,  # default root logger so that dask logging is forwarded
        path=output_dir,
        logfile="protein-prep.log",
        stdout=True,
        level=inputs.loglevel,
    ).getLogger()

    logger.info(f"Running protein prep with inputs: {inputs}")
    logger.info(f"Dumping input schema to {output_dir / 'inputs.json'}")

    inputs.to_json_file(output_dir / "protein_prep.json")

    if inputs.use_dask:
        dask_client = make_dask_client_meta(
            inputs.dask_type,
            loglevel=inputs.loglevel,
            n_workers=inputs.dask_n_workers,
        )
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
        complex_target = Complex.from_pdb(
            inputs.pdb_file,
            target_kwargs={"target_name": inputs.pdb_file.stem},
            ligand_kwargs={"compound_name": f"{inputs.pdb_file.stem}_ligand"},
        )
        complexes = [complex_target]

    else:
        raise ValueError(
            "Must specify either fragalysis_dir, structure_dir or pdb_file"
        )

    logger.info(f"Loaded {len(complexes)} complexes")

    if not inputs.seqres_yaml:
        logger.info("No seqres yaml specified")
    else:
        logger.info(f"Using seqres yaml: {inputs.seqres_yaml}")
    if inputs.align:
        # load reference structure
        logger.info(f"Loading and aligning to reference structure: {inputs.align}")
        align_struct = inputs.align
    else:
        logger.info("No reference structure specified, using canonical structure")
        align_struct = master_structures[inputs.target]

    ref_complex = Complex.from_pdb(
        align_struct,
        target_kwargs={"target_name": "ref"},
        ligand_kwargs={"compound_name": "ref_ligand"},
    )

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
        inputs=complexes,
        use_dask=inputs.use_dask,
        dask_client=dask_client,
        cache_dir=inputs.cache_dir,
    )

    logger.info(f"Prepped {len(prepped_complexes)} complexes")

    logger.info(f"Writing prepped complexes to {inputs.output_dir}")
    prepper.cache(prepped_complexes, inputs.output_dir)

    if inputs.save_to_cache and inputs.cache_dir is not None:
        logger.info(f"Writing prepped complexes to global cache {inputs.cache_dir}")
        prepper.cache(prepped_complexes, inputs.cache_dir)

    logger.info("Done")
