from pydantic import BaseModel, Field, root_validator
from typing import Optional
from asapdiscovery.data.postera.manifold_data_validation import TargetTags


class LargeScaleDockingInputs(BaseModel):
    filename: Optional[str] = Field(
        None, description="Path to a molecule file containing query ligands."
    )
    frag_dir: Optional[str] = Field(
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
    target: TargetTags = Field(None, description="The target to dock against.")
    write_final_sdf: bool = Field(
        default=False,
        description="Whether to write the final docked poses to an SDF file.",
    )

    dask_client: Optional[distributed.Client] = Field(
        description="Dask client to use for parallelism."
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
        frag_dir = values.get("frag_dir")
        structure_dir = values.get("structure_dir")
        postera = values.get("postera")
        postera_upload = values.get("postera_upload")
        postera_molset_name = values.get("postera_molset_name")
        du_cache = values.get("du_cache")
        target = values.get("target")

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

        if frag_dir and structure_dir:
            raise ValueError("Cannot specify both frag_dir and structure_dir.")

        if not frag_dir and not structure_dir:
            raise ValueError("Must specify either frag_dir or structure_dir.")

        return values

    @validator("dask_client")
    @classmethod
    def spawn_dask_client(cls, v):
        if v is None:
            return distributed.Client()
        else:
            return v


def large_scale_docking(
    postera: bool,
    postera_upload: bool,
    target: TargetTags,
    n_select: int,
    write_final_sdf: bool,
    dask_type: str,
    filename: Optional[str | Path] = None,
    frag_dir: Optional[str | Path] = None,
    structure_dir: Optional[str | Path] = None,
    postera_molset_name: Optional[str] = None,
    du_cache: Optional[str | Path] = None,
):
    """
    Run large scale docking on a set of ligands, against a single target.
    """

    # Code for large scale docking
    inputs = LargeScaleDockingInputs(
        filename=filename,
        frag_dir=frag_dir,
        structure_dir=structure_dir,
        postera=postera,
        postera_upload=postera_upload,
        postera_molset_name=postera_molset_name,
        du_cache=du_cache,
        target=target,
        write_final_sdf=write_final_sdf,
        dask_type=dask_type,
    )

    dask_client = dask_client_from_type(inputs.dask_type)

    if postera:
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

    # load fragalysis and ligands
    fragalysis = FragalysisFactory.from_dir(inputs.frag_dir)
    complexes = fragalysis.load()  # TODO: factory pattern for loading complexes
    prepper = ProteinPrepper()
    prepped_complexes = prepper.prep(
        complexes, du_cache_dir=inputs.du_cache, use_dask=True, dask_client=dask_client
    )  # TODO: fix + caching

    # define selector and select pairs
    selector = MCSSelector()
    pairs = selector.select(
        query_ligands,
        prepped_complexes,
        n_select=10,
        use_dask=True,
        dask_client=dask_client,
    )  # TODO: add dask parallelism to selector

    # dock pairs
    docker = POSITDocker()
    results = docker.dock(
        pairs,
        use_dask=True,
        dask_client=inputs.dask_client,
        write_files=True,
        output_dir="docking_results",
    )

    # write results to dataframe
    result_df = make_df_from_docking_results(results)
    result_df = rename_output_columns_for_manifold(
        result_df,
        target,
        [DockingResultCols],
        manifold_validate=True,
        drop_non_output=True,
    )  # TODO:  we can make this nicer for sure, this function is ugly AF
    result_df.to_csv("docking_results.csv", index=False)

    if postera_upload:
        postera_uploader = PosteraUploader(
            settings=settings, molecule_set_name=inputs.postera_molset_name
        )  # TODO: make this more compact wrapper for postera uploader
        postera_uploader.upload(df)

    if inputs.write_final_sdf:
        write_multi_sdf(results, "docking_results.sdf")
