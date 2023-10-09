import click

# from pathlib import Path
from typing import Optional

from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.data.dask_utils import DaskType, dask_client_from_type
from asapdiscovery.docking.workflows.large_scale_docking import large_scale_docking


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
    type=str,
    help="The target to dock against.",
)
@click.option(
    "--n-select",
    type=int,
    default=10,
    help="The number of targets to dock each ligand against, sorted by MCS",
)
@click.option(
    "--write-final-sdf",
    is_flag=True,
    default=True,
    help="Whether to write the final docked poses to an SDF file.",
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
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="Path to a directory where a design unit cache should be generated",
)
def large_scale(
    postera: bool,
    postera_upload: bool,
    target: TargetTags,
    n_select: int,
    write_final_sdf: bool,
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

    large_scale_docking(
        postera=postera,
        postera_upload=postera_upload,
        target=target,
        n_select=n_select,
        write_final_sdf=write_final_sdf,
        dask_type=dask_type,
        filename=ligands,
        fragalysis_dir=fragalysis_dir,
        structure_dir=structure_dir,
        postera_molset_name=postera_molset_name,
        du_cache=du_cache,
        gen_du_cache=gen_du_cache,
    )


if __name__ == "__main__":
    cli()
