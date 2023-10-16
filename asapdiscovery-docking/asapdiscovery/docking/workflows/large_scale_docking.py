from pathlib import Path
from typing import Optional

from asapdiscovery.data.dask_utils import DaskType, dask_client_from_type
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
from asapdiscovery.docking.docking import DockingResultCols
from asapdiscovery.docking.docking_v2 import POSITDocker
from asapdiscovery.docking.scorer_v2 import (
    ChemGauss4Scorer,
    GATScorer,
    MetaScorer,
    SchnetScorer,
)
from asapdiscovery.modeling.protein_prep_v2 import ProteinPrepper
from pydantic import BaseModel, Field, root_validator, validator


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
        default=False,
        description="Whether to write the final docked poses to an SDF file.",
    )

    dask_type: DaskType = Field(
        DaskType.LOCAL, description="Dask client to use for parallelism."
    )

    class Config:
        arbitrary_types_allowed = True

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

        if postera_upload and not postera:
            raise ValueError("Cannot specify postera_upload without postera.")

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
    Run large scale docking on a set of ligands, against a single target.
    """
    dask_client = dask_client_from_type(inputs.dask_type)

    if inputs.postera:
        # load postera
        postera_settings = PosteraSettings()
        postera = PosteraFactory(
            settings=postera_settings, molecule_set_name=inputs.postera_molset_name
        )
        query_ligands = postera.pull()
    else:
        # load from file
        molfile = MolFileFactory.from_file(inputs.filename)
        query_ligands = molfile.ligands

    # load complexes from a directory or from fragalysis
    if inputs.structure_dir:
        structure_factory = StructureDirFactory.from_dir(inputs.structure_dir)
        complexes = structure_factory.load(use_dask=True, dask_client=dask_client)
    else:
        fragalysis = FragalysisFactory.from_dir(inputs.fragalysis_dir)
        complexes = fragalysis.load(use_dask=True, dask_client=dask_client)

    prepper = ProteinPrepper(du_cache=inputs.du_cache)
    prepped_complexes = prepper.prep(complexes, use_dask=True, dask_client=dask_client)

    if inputs.gen_du_cache and not inputs.du_cache:
        prepper.cache(prepped_complexes, inputs.gen_du_cache)

    # define selector and select pairs
    selector = MCSSelector()
    pairs = selector.select(
        query_ligands,
        prepped_complexes,
        n_select=10,
        use_dask=False,  # TODO fix dask here
        dask_client=None,
    )

    # dock pairs
    docker = POSITDocker()
    results = docker.dock(
        pairs,
        use_dask=True,
        dask_client=dask_client,
    )
    POSITDocker.write_docking_files(results, Path("docking_results"))

    # score results
    scorer = MetaScorer(
        scorers=[
            ChemGauss4Scorer(),
            GATScorer.from_latest_by_target(inputs.target),
            SchnetScorer.from_latest_by_target(inputs.target),
        ]
    )

    scores_df = scorer.score(
        results, use_dask=True, dask_client=dask_client, return_df=True
    )
    scores_df.to_csv("docking_scores.csv", index=False)

    result_df = rename_output_columns_for_manifold(
        result_df,
        inputs.target,
        [DockingResultCols],
        manifold_validate=True,
        drop_non_output=True,
    )  # TODO:  we can make this nicer for sure, this function is ugly AF

    result_df.to_csv("docking_results_final.csv", index=False)

    if inputs.postera_upload:
        postera_uploader = PosteraUploader(
            settings=inputs.settings, molecule_set_name=inputs.postera_molset_name
        )  # TODO: make this more compact wrapper for postera uploader
        postera_uploader.upload(result_df)

    if inputs.write_final_sdf:
        write_ligands_to_multi_sdf(
            "docking_results.sdf", [r.posed_ligand for r in results]
        )
