"""
Base class for docking workflows
"""
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
from asapdiscovery.data.selectors.mcs_selector import MCSSelector
from asapdiscovery.data.services_config import PosteraSettings
from asapdiscovery.data.utils import check_empty_dataframe
from asapdiscovery.docking.docking_data_validation import (
    DockingResultColsV2 as DockingResultCols,
)
from asapdiscovery.docking.docking_v2 import POSITDocker
from asapdiscovery.docking.scorer_v2 import ChemGauss4Scorer, MetaScorer, MLModelScorer
from asapdiscovery.ml.models.ml_models import ASAPMLModelRegistry
from asapdiscovery.modeling.protein_prep_v2 import CacheType, ProteinPrepper
from distributed import Client
from pydantic import BaseModel, Field, PositiveInt, root_validator, validator

# TODO: delete this and use the one from the other branch once it's merged
class DockingInputs(BaseModel):
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

    n_select: PositiveInt = Field(
        5, description="Number of targets to dock each ligand against, sorted by MCS"
    )

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


class DockingWorkflowBase(BaseModel):
    inputs: DockingInputs
    type: str = Field("DockingWorkflowBase", description="The type of workflow.")

    def run_workflow(self):
        self._output_dir()
        self._logging()
        self._initialization()
        self._dask_client()
        self._loading_inputs()
        self._prep()
        self._postprocessing()
        self._uploading()
        self._finalization()
        for step in registered_steps:
            step(self)

    def _output_dir(self):
        if self.inputs.output_dir.exists():
            rmtree(self.inputs.output_dir)
        self.inputs.output_dir.mkdir()

    def _logging(self):
        self.logger = FileLogger(
            self.inputs.logname,
            path=self.inputs.output_dir,
            stdout=True,
            level=self.inputs.loglevel,
        ).getLogger()

    def _initialization(self):
        self.logger.info(f"Running {self.type} with inputs: {self.inputs}")
        self.logger.info(
            f"Dumping input schema to {self.inputs.output_dir / 'inputs.json'}"
        )

        self.inputs.to_json_file(
            self.inputs.output_dir / "large_scale_docking_inputs.json"
        )

    def _dask_client(self):
        if self.inputs.use_dask:
            set_dask_config()
            self.logger.info(
                f"Using dask for parallelism of type: {self.inputs.dask_type}"
            )
            dask_cluster = dask_cluster_from_type(self.inputs.dask_type)

            if self.inputs.dask_type.is_lilac():
                self.logger.info("Lilac HPC config selected, setting adaptive scaling")
                dask_cluster.adapt(
                    minimum=10,
                    maximum=self.inputs.dask_cluster_max_workers,
                    wait_count=10,
                    interval="1m",
                )
                self.logger.info(
                    f"Estimating {self.inputs.dask_cluster_n_workers} workers"
                )
                dask_cluster.scale(self.inputs.dask_cluster_n_workers)

            self.dask_client = Client(dask_cluster)
            self.logger.info(f"Using dask client: {self.dask_client}")
            self.logger.info(f"Using dask cluster: {dask_cluster}")
            self.logger.info(
                f"Dask client dashboard: {self.dask_client.dashboard_link}"
            )

        else:
            self.dask_client = None

    def _loading_inputs(self):
        # make a directory to store intermediate CSV results
        data_intermediates = Path(self.inputs.output_dir / "data_intermediates")
        data_intermediates.mkdir(exist_ok=True)

        if self.inputs.postera:
            # load postera
            self.logger.info(
                f"Loading Postera database molecule set {self.inputs.postera_molset_name}"
            )
            postera_settings = PosteraSettings()
            postera = PosteraFactory(
                settings=postera_settings,
                molecule_set_name=self.inputs.postera_molset_name,
            )
            query_ligands = postera.pull()
        else:
            # load from file
            self.logger.info(f"Loading ligands from file: {self.inputs.filename}")
            molfile = MolFileFactory.from_file(self.inputs.filename)
            query_ligands = molfile.ligands

        # load complexes from a directory, from fragalysis or from a pdb file
        if self.inputs.structure_dir:
            self.logger.info(
                f"Loading structures from directory: {self.inputs.structure_dir}"
            )
            structure_factory = StructureDirFactory.from_dir(self.inputs.structure_dir)
            complexes = structure_factory.load(
                use_dask=self.inputs.use_dask, dask_client=self.dask_client
            )
        elif self.inputs.fragalysis_dir:
            self.logger.info(
                f"Loading structures from fragalysis: {self.inputs.fragalysis_dir}"
            )
            fragalysis = FragalysisFactory.from_dir(self.inputs.fragalysis_dir)
            complexes = fragalysis.load(
                use_dask=self.inputs.use_dask, dask_client=self.dask_client
            )

        elif self.inputs.pdb_file:
            self.logger.info(f"Loading structures from pdb: {self.inputs.pdb_file}")
            complex = Complex.from_pdb(
                self.inputs.pdb_file,
                target_kwargs={"target_name": self.inputs.pdb_file.stem},
                ligand_kwargs={"compound_name": f"{self.inputs.pdb_file.stem}_ligand"},
            )
            complexes = [complex]

        else:
            raise ValueError(
                "Must specify either fragalysis_dir, structure_dir or pdb_file"
            )

        n_query_ligands = len(query_ligands)
        self.logger.info(f"Loaded {n_query_ligands} query ligands")
        n_complexes = len(complexes)
        self.logger.info(f"Loaded {n_complexes} complexes")
