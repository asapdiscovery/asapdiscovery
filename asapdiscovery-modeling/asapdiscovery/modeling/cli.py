from typing import TYPE_CHECKING, Optional

import click
from asapdiscovery.cli.cli_args import (
    dask_args,
    fragalysis_dir,
    input_json,
    output_dir,
    pdb_file,
    save_to_cache,
    structure_dir,
    target,
)
from asapdiscovery.data.util.dask_utils import DaskFailureMode, DaskType

if TYPE_CHECKING:
    from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags


@click.group()
def modeling():
    pass


@modeling.command()
@target
@click.option(
    "--align",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="Path to a reference structure to align to",
)
@click.option(
    "--ref-chain",
    type=str,
    default="A",
    help="Chain ID to align to",
)
@click.option(
    "--active-site-chain",
    type=str,
    default="A",
    help="Active site chain ID to align to",
)
@click.option(
    "--seqres-yaml",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="Path to a seqres yaml file to mutate to, if not specified will use the default for the target",
)
@click.option(
    "--loop-db",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="Path to a loop database to use for prepping",
)
@click.option(
    "--oe-active-site-residue",
    type=str,
    help="OE formatted string of active site residue to use if not ligand bound",
)
@pdb_file
@fragalysis_dir
@structure_dir
@click.option(
    "--cache-dir",
    help="The path to cached prepared complexes which can be used again.",
    type=click.Path(resolve_path=True, exists=True, file_okay=False, dir_okay=True),
)
@save_to_cache
@dask_args
@output_dir
@input_json
def protein_prep(
    target: "TargetTags",
    align: Optional[str] = None,
    ref_chain: Optional[str] = None,
    active_site_chain: Optional[str] = None,
    seqres_yaml: Optional[str] = None,
    loop_db: Optional[str] = None,
    oe_active_site_residue: Optional[str] = None,
    pdb_file: Optional[str] = None,
    fragalysis_dir: Optional[str] = None,
    structure_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    save_to_cache: bool = True,
    use_dask: bool = False,
    dask_type: DaskType = DaskType.LOCAL,
    dask_failure_mode: DaskFailureMode = DaskFailureMode.SKIP,
    output_dir: str = "output",
    input_json: Optional[str] = None,
):
    """
    Run protein prep on a set of structures.
    """
    from asapdiscovery.modeling.workflows.protein_prep import (
        ProteinPrepInputs,
        protein_prep_workflow,
    )

    if input_json is not None:
        print("Loading inputs from json file... Will override all other inputs.")
        inputs = ProteinPrepInputs.from_json_file(input_json)

    else:
        inputs = ProteinPrepInputs(
            target=target,
            align=align,
            ref_chain=ref_chain,
            active_site_chain=active_site_chain,
            seqres_yaml=seqres_yaml,
            loop_db=loop_db,
            oe_active_site_residue=oe_active_site_residue,
            pdb_file=pdb_file,
            fragalysis_dir=fragalysis_dir,
            structure_dir=structure_dir,
            cache_dir=cache_dir,
            save_to_cache=save_to_cache,
            use_dask=use_dask,
            dask_type=dask_type,
            dask_failure_mode=dask_failure_mode,
            output_dir=output_dir,
        )

    protein_prep_workflow(inputs)


if __name__ == "__main__":
    modeling()
