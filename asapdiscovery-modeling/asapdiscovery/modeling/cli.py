from typing import Optional

import click
from asapdiscovery.data.dask_utils import DaskType
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.modeling.workflows.protein_prep import (
    ProteinPrepInputs,
    protein_prep,
)

from asapdiscovery.cli.cli_args import (
    target,
    dask_args,
    output_dir,
    input_json,
    fragalysis_dir,
    structure_dir,
    pdb_file,
)


@click.group()
def cli():
    pass


@cli.command()
@target
@pdb_file
@fragalysis_dir
@structure_dir
@gen_cache
@dask_args
@output_dir
@input_json
def prep(
    target: TargetTags,
    pdb_file: Optional[str] = None,
    fragalysis_dir: Optional[str] = None,
    structure_dir: Optional[str] = None,
    gen_du_cache: Optional[str] = None,
    use_dask: bool = False,
    dask_type: DaskType = DaskType.LOCAL,
    output_dir: str = "output",
    input_json: Optional[str] = None,
):
    """
    Run large scale docking on a set of ligands, against a set of targets.
    """

    if input_json is not None:
        print("Loading inputs from json file... Will override all other inputs.")
        inputs = ProteinPrepInputs.from_json_file(input_json)

    else:
        inputs = ProteinPrepInputs(
            target=target,
            fragalysis_dir=fragalysis_dir,
            structure_dir=structure_dir,
            pdb_file=pdb_file,
            gen_du_cache=gen_du_cache,
            use_dask=use_dask,
            dask_type=dask_type,
            output_dir=output_dir,
        )

    protein_prep(inputs)


if __name__ == "__main__":
    cli()
