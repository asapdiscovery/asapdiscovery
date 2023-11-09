from typing import Optional

import click
from asapdiscovery.cli.cli_args import (
    cache_type,
    dask_args,
    fragalysis_dir,
    gen_cache_w_default,
    input_json,
    output_dir,
    pdb_file,
    structure_dir,
    target,
)
from asapdiscovery.data.dask_utils import DaskType
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.modeling.protein_prep_v2 import CacheType
from asapdiscovery.modeling.workflows.protein_prep import (
    ProteinPrepInputs,
    protein_prep_workflow,
)


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
    help="Chain ID to align to",
)
@click.option(
    "--active-site-chain",
    type=str,
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
@gen_cache_w_default
@cache_type
@dask_args
@output_dir
@input_json
def protein_prep(
    target: TargetTags,
    align: Optional[str] = None,
    ref_chain: Optional[str] = None,
    active_site_chain: Optional[str] = None,
    seqres_yaml: Optional[str] = None,
    loop_db: Optional[str] = None,
    oe_active_site_residue: Optional[str] = None,
    pdb_file: Optional[str] = None,
    fragalysis_dir: Optional[str] = None,
    structure_dir: Optional[str] = None,
    gen_cache: Optional[str] = "prepped_structure_cache",
    cache_type: Optional[list[str]] = [CacheType.DesignUnit],
    use_dask: bool = False,
    dask_type: DaskType = DaskType.LOCAL,
    output_dir: str = "output",
    input_json: Optional[str] = None,
):
    """
    Run protein prep on a set of structures.
    """

    if input_json is not None:
        print("Loading inputs from json file... Will override all other inputs.")
        inputs = ProteinPrepInputs.from_json_file(input_json)

    else:
        print(output_dir)
        print(gen_cache)
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
            gen_cache=gen_cache,
            cache_type=cache_type,
            use_dask=use_dask,
            dask_type=dask_type,
            output_dir=output_dir,
        )

    protein_prep_workflow(inputs)


if __name__ == "__main__":
    modeling()
