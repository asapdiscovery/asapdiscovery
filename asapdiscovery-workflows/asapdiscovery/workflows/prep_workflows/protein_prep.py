import logging
from pathlib import Path
from typing import Optional, Union

from asapdiscovery.data.metadata.resources import master_structures
from asapdiscovery.data.readers.structure_dir import StructureDirFactory
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.sequence import seqres_by_target
from asapdiscovery.data.services.fragalysis.fragalysis_reader import FragalysisFactory
from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags
from asapdiscovery.data.services.postera.postera_factory import PosteraFactory
from asapdiscovery.data.util.dask_utils import (
    DaskType,
    dask_cluster_from_type,
    set_dask_config,
)
from asapdiscovery.data.util.logging import FileLogger
from asapdiscovery.modeling.protein_prep import ProteinPrepper
from distributed import Client
from pydantic import BaseModel, Field, PositiveInt, root_validator


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
    alchemy_compounds_sdf : Optional[Path]
        Path to an SDF file to find a suitable reference for and then prep it
    alchemy_compounds_postera : Optional[str]
        Name of Postera MoleculeSet to find a suitable reference for and then prep it
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
    alchemy_compounds_sdf: Optional[Path] = Field(
        None,
        description="Path to an SDF file to find a suitable reference for and then prep it",
    )
    alchemy_compounds_postera: Optional[str] = Field(
        None,
        description="Name of Postera MoleculeSet to find a suitable reference for and then prep it",
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

    dask_cluster_n_workers: PositiveInt = Field(
        10,
        description="Number of workers to use as initial guess for Lilac dask cluster",
    )

    dask_cluster_max_workers: PositiveInt = Field(
        40, description="Maximum number of workers to use for Lilac dask cluster"
    )

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

    # if we're finding a reference for a set of compounds for asap-alchemy,
    # check that a structures dir is defined
    if (
        inputs.alchemy_compounds_sdf
        and not inputs.structure_dir
        or inputs.alchemy_compounds_postera
        and not inputs.structure_dir
    ):
        raise ValueError(
            "If specifying alchemy_compounds_*, must specify structure_dir"
        )
    logger.info(f"Running protein prep with inputs: {inputs}")
    logger.info(f"Dumping input schema to {output_dir / 'inputs.json'}")

    inputs.to_json_file(output_dir / "protein_prep.json")

    if inputs.use_dask:
        logger.info(f"Using dask for parallelism of type: {inputs.dask_type}")
        set_dask_config()
        dask_cluster = dask_cluster_from_type(
            inputs.dask_type, loglevel=inputs.loglevel
        )

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
        dask_client.forward_logging(level=inputs.loglevel)
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
        logger.info(
            f"No seqres yaml specified, selecting based on target: {inputs.target}"
        )
        inputs.seqres_yaml = seqres_by_target(inputs.target)

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
    if inputs.alchemy_compounds_sdf:
        logger.info(
            f"Finding a suitable reference for {inputs.alchemy_compounds_sdf} in folder: {inputs.structure_dir}"
        )
        input_mols = [
            Chem.MolToSmiles(mol)
            for mol in Chem.SDMolSupplier(str(inputs.alchemy_compounds_sdf))
        ]
        prepped_complexes = select_reference_for_compounds(
            input_mols, complexes, prepper, inputs, dask_client, logger
        )
    elif inputs.alchemy_compounds_postera:
        logger.info(
            f"Finding a suitable reference for MoleculeSet {inputs.alchemy_compounds_postera} in folder: {inputs.structure_dir}"
        )
        postera_factory = PosteraFactory(
            molecule_set_name=str(inputs.alchemy_compounds_postera)
        )
        input_mols = [mol.smiles for mol in postera_factory.pull()]
        prepped_complexes = select_reference_for_compounds(
            input_mols, complexes, prepper, inputs, dask_client, logger
        )
    else:
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


from asapdiscovery.data.operators.selectors.mcs_selector import MCSSelector
from asapdiscovery.data.schema.ligand import Ligand
from rdkit import Chem


def select_reference_for_compounds(
    input_mols, complexes, prepper, inputs, dask_client, logger
):
    """
    From a collection of ligands and a list of `Complex`es, return the `Complex` that is the most similar to
    the largest of the query ligands.

    Parameters
    ----------
    input_mols : Path
        Path to an SDF file containing molecules to query
    complexes : list
        List of `Complex`es to find the most similar `Complex` in
    prepper: ProteinPrepper
        Object to prep `Complex`es with
    inputs : ProteinPrepInputs
        ..
    dask_client :
        ..
    logger :
        ..

    Returns
    -------
    Schema Complex
        `Complex` that was found to be most similar to the largest compound in `input_mol`
    """
    # find the largest molecule in terms of surface area
    input_mols = [Chem.MolFromSmiles(smi) for smi in input_mols]
    largest_compound = "CCO"  # stick to SMILES as base variable for easier logging
    for mol in input_mols:
        if mol.GetNumAtoms() > Chem.MolFromSmiles(largest_compound).GetNumAtoms():
            largest_compound = Chem.MolToSmiles(mol)

    # find that largest ligand's `n` closest reference
    selector = MCSSelector()
    pairs = selector.select(
        [Ligand.from_smiles(largest_compound, compound_name="query_compound")],
        complexes,
        n_select=10,  # if we run out of references we can either
        # increase this number or start taking from e.g. the second-largest compound
        # we can also use this point to build multi-ref FECs if we need to
    )
    reference_complexes = []
    for pair in pairs:
        target_name = pair.complex.target.target_name
        logger.info(f"Found a match with {target_name}; attempting prep")
        reference_complexes.append(target_name)
        prepped_complex = prepper.prep(
            inputs=[pair.complex],
            use_dask=inputs.use_dask,
            dask_client=dask_client,
            cache_dir=inputs.cache_dir,
        )[0]
        logger.info(f"Checking that the prepped complex is compatible with OpenMM")
        prepped_complex.target.to_pdb_file(
            f"{target_name}.pdb"
        )  # @HMO: this should be in a tmpdir?
        # @HMO: can we the below more elegantly? I don't really like this setup with the below ValueError.
        try:
            prepped_simulation = create_protein_only_system(f"{target_name}.pdb")
        except Exception as e:
            prepped_simulation = False
            logger.warn(
                f"Reference {target_name} is not compatible with OpenMM:\n{e}\n\n ..trying next reference.."
            )
            pass
        if prepped_simulation:
            logger.info(f"{target_name} is compatible")
            smiles_reference, smiles_query, similarity = get_similarity(
                pair.complex.ligand.smiles, largest_compound
            )
            logger.info(
                f"Using reference {target_name} such that\nReference compound: {smiles_reference}\nQuery compound from FECs set: {smiles_query}\nSimilarity: {similarity}"
            )
            if similarity < 0.2:
                logger.warn(
                    f"Low similarity with reference crystal pose ligand: {similarity}, check prep/docking results carefully"
                )
            return [prepped_complex]
        else:
            pass
    raise ValueError(  # if we reach this point then we've failed to find a reference
        f"No references were found for query ligand {largest_compound} in references: {reference_complexes}"
    )


#### BELOW SHOULD BE SOMEWHERE ELSE?
import openmm
from openff.toolkit import ForceField, Topology
from openmm import unit as openmm_unit


### SOMETHING GOING WRONG HERE - NO BOX?!
def create_protein_only_system(pdb_path):
    # attempt to make an OpenMM system with the prepped protein.
    # follows https://docs.openforcefield.org/en/latest/examples/openforcefield/openff-toolkit/toolkit_showcase/toolkit_showcase.html
    top = Topology.from_pdb(pdb_path)
    sage_ff14sb = ForceField("openff-2.1.0.offxml", "ff14sb_off_impropers_0.0.3.offxml")
    interchange = sage_ff14sb.create_interchange(top)

    # Under the hood, this creates *OpenMM* `System` and `Topology` objects, then combines them together
    simulation = interchange.to_openmm_simulation(
        integrator=openmm.LangevinIntegrator(
            300 * openmm_unit.kelvin,
            1 / openmm_unit.picosecond,
            0.002 * openmm_unit.picoseconds,
        )
    )

    return simulation


from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def get_similarity(smiles_a, smiles_b):
    # returns ECFP6 tanimoto similarity between two input SMILES
    radius = 3  # ECFP6 because of diameter instead of radius
    simi = DataStructs.FingerprintSimilarity(
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles_a), radius),
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles_b), radius),
    )

    return smiles_a, smiles_b, round(simi, 2)


#####
