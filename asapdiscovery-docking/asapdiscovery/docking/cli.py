# from pathlib import Path
from typing import Optional

import click
from asapdiscovery.data.dask_utils import DaskType, dask_client_from_type
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.docking.workflows.large_scale_docking import (
    LargeScaleDockingInputs,
    large_scale_docking,
)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--postera",
    is_flag=True,
    default=False,
    help="Whether to download complexes from Postera.",
)
@click.option(
    "--postera-upload",
    is_flag=True,
    default=False,
    help="Whether to upload the results to Postera.",
)
@click.option(
    "--target",
    type=click.Choice(TargetTags.get_values(), case_sensitive=True),
    help="The target to dock against.",
    required=True,
)
@click.option(
    "--n-select",
    type=int,
    default=10,
    help="The number of targets to dock each ligand against, sorted by MCS",
)
@click.option(
    "--top-n",
    type=int,
    default=500,
    help="The maximum number of docking results to return, ordered by docking score",
)
@click.option(
    "--no-write-final-sdf",
    is_flag=True,
    default=True,
    help="Whether to write the final docked poses to an SDF file.",
)
@click.option(
    "--use-dask",
    is_flag=True,
    default=False,
    help="Whether to use dask for parallelism.",
)
@click.option(
    "--dask-type",
    type=DaskType,
    default=DaskType.LOCAL,
    help="The type of dask cluster to use. Can be 'local' or 'lilac-cpu'.",
)
@click.option(
    "-l",
    "--ligands",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="Path to a file containing a list of complexes to dock.",
)
@click.option(
    "--fragalysis-dir",
    type=click.Path(resolve_path=True, exists=True, file_okay=False, dir_okay=True),
    help="Path to a directory containing fragments to dock.",
)
@click.option(
    "--structure-dir",
    type=click.Path(resolve_path=True, exists=True, file_okay=False, dir_okay=True),
    help="Path to a directory containing structures to dock instead of a full fragalysis database.",
)
@click.option(
    "--postera-molset-name",
    type=str,
    default=None,
    help="The name of the molecule set to pull from and upload to.",
)
@click.option(
    "--du-cache",
    type=click.Path(resolve_path=True, exists=True, file_okay=False, dir_okay=True),
    help="Path to a directory where design units are cached",
)
@click.option(
    "--gen-du-cache",
    type=click.Path(resolve_path=True, exists=False, file_okay=False, dir_okay=True),
    help="Path to a directory where a design unit cache should be generated",
)
def large_scale(
    postera: bool,
    postera_upload: bool,
    target: TargetTags,
    n_select: int,
    top_n: int,
    write_final_sdf: bool,
    use_dask: bool,
    dask_type: str,
    ligands: Optional[str] = None,
    fragalysis_dir: Optional[str] = None,
    structure_dir: Optional[str] = None,
    postera_molset_name: Optional[str] = None,
    du_cache: Optional[str] = None,
    gen_du_cache: Optional[str] = None,
):
    """
    Run large scale docking on a set of ligands, against a set of targets.
    """

    inputs = LargeScaleDockingInputs(
        postera=postera,
        postera_upload=postera_upload,
        target=target,
        n_select=n_select,
        top_n=top_n,
        write_final_sdf=write_final_sdf,
        use_dask=use_dask,
        dask_type=dask_type,
        filename=ligands,
        fragalysis_dir=fragalysis_dir,
        structure_dir=structure_dir,
        postera_molset_name=postera_molset_name,
        du_cache=du_cache,
        gen_du_cache=gen_du_cache,
    )

    large_scale_docking(inputs)


if __name__ == "__main__":
    cli()
