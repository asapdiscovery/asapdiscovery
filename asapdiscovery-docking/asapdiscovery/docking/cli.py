from typing import Optional

import click
from asapdiscovery.data.dask_utils import DaskType
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.docking.workflows.large_scale_docking import (
    LargeScaleDockingInputs,
    large_scale_docking,
)

from asapdiscovery.cli.cli_args import (
    postera_args,
    ml_scorer,
    target,
    dask_args,
    output_dir,
    input_json,
    ligands,
    structure_and_cache_params,
)


@click.group()
def cli():
    pass


@cli.command()
@target
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
    "--posit-confidence-cutoff",
    type=float,
    default=0.7,
    help="The confidence cutoff for POSIT results to be considered",
)
@ligands
@postera_args
@dask_args
@output_dir
@input_json
@structure_and_cache_params
@ml_scorer
def large_scale(
    target: TargetTags,
    n_select: int = 10,
    top_n: int = 500,
    posit_confidence_cutoff: float = 0.7,
    ligands: Optional[str] = None,
    postera: bool = False,
    postera_molset_name: Optional[str] = None,
    postera_upload: bool = False,
    use_dask: bool = False,
    dask_type: DaskType = DaskType.LOCAL,
    output_dir: str = "output",
    input_json: Optional[str] = None,
    pdb_file: Optional[str] = None,
    fragalysis_dir: Optional[str] = None,
    structure_dir: Optional[str] = None,
    du_cache: Optional[str] = None,
    gen_du_cache: Optional[str] = None,
    ml_scorer: Optional[list[str]] = None,
):
    """
    Run large scale docking on a set of ligands, against a set of targets.
    """

    if input_json is not None:
        print("Loading inputs from json file... Will override all other inputs.")
        inputs = LargeScaleDockingInputs.from_json_file(input_json)

    else:
        inputs = LargeScaleDockingInputs(
            postera=postera,
            postera_upload=postera_upload,
            target=target,
            n_select=n_select,
            top_n=top_n,
            use_dask=use_dask,
            dask_type=dask_type,
            posit_confidence_cutoff=posit_confidence_cutoff,
            filename=ligands,
            pdb_file=pdb_file,
            fragalysis_dir=fragalysis_dir,
            structure_dir=structure_dir,
            postera_molset_name=postera_molset_name,
            du_cache=du_cache,
            gen_du_cache=gen_du_cache,
            ml_scorers=ml_scorer,
            output_dir=output_dir,
        )

    large_scale_docking(inputs)


if __name__ == "__main__":
    cli()
