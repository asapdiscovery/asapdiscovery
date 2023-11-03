from typing import Optional

import click
from asapdiscovery.cli.cli_args import (
    cache_dir,
    cache_type,
    dask_args,
    fragalysis_dir,
    gen_cache,
    input_json,
    ligands,
    ml_scorer,
    output_dir,
    pdb_file,
    postera_args,
    structure_dir,
    target,
    md,
    md_steps,
    md_openmm_platform,
)
from asapdiscovery.data.dask_utils import DaskType
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.simulation.simulate import OpenMMPlatform
from asapdiscovery.docking.workflows.large_scale_docking import (
    LargeScaleDockingInputs,
    large_scale_docking_workflow,
)

from asapdiscovery.docking.workflows.small_scale_docking import (
    SmallScaleDockingInputs,
    small_scale_docking_workflow,
)


@click.group()
def docking():
    pass


@docking.command()
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
@pdb_file
@fragalysis_dir
@structure_dir
@gen_cache
@cache_dir
@cache_type
@dask_args
@output_dir
@input_json
@ml_scorer
def large_scale(
    target: TargetTags,
    n_select: int = 5,
    top_n: int = 500,
    posit_confidence_cutoff: float = 0.7,
    ligands: Optional[str] = None,
    postera: bool = False,
    postera_molset_name: Optional[str] = None,
    postera_upload: bool = False,
    pdb_file: Optional[str] = None,
    fragalysis_dir: Optional[str] = None,
    structure_dir: Optional[str] = None,
    gen_cache: Optional[str] = None,
    cache_dir: Optional[str] = None,
    cache_type: Optional[str] = None,
    output_dir: str = "output",
    input_json: Optional[str] = None,
    use_dask: bool = False,
    dask_type: DaskType = DaskType.LOCAL,
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
            cache_dir=cache_dir,
            gen_cache=gen_cache,
            cache_type=cache_type,
            ml_scorers=ml_scorer,
            output_dir=output_dir,
        )

    large_scale_docking_workflow(inputs)


@docking.command()
@target
@click.option(
    "--posit-confidence-cutoff",
    type=float,
    default=0.1,
    help="The confidence cutoff for POSIT results to be considered",
)
@ligands
@postera_args
@pdb_file
@fragalysis_dir
@structure_dir
@gen_cache
@cache_dir
@cache_type
@dask_args
@output_dir
@input_json
@ml_scorer
@md
@md_steps
@md_openmm_platform
def small_scale(
    target: TargetTags,
    posit_confidence_cutoff: float = 0.1,
    ligands: Optional[str] = None,
    postera: bool = False,
    postera_molset_name: Optional[str] = None,
    postera_upload: bool = False,
    pdb_file: Optional[str] = None,
    fragalysis_dir: Optional[str] = None,
    structure_dir: Optional[str] = None,
    gen_cache: Optional[str] = None,
    cache_dir: Optional[str] = None,
    cache_type: Optional[str] = None,
    output_dir: str = "output",
    input_json: Optional[str] = None,
    use_dask: bool = False,
    dask_type: DaskType = DaskType.LOCAL,
    ml_scorer: Optional[list[str]] = None,
    md: bool = False,
    md_steps: int = 2500000,  # 10 ns @ 4.0 fs timestep
    md_openmm_platform: OpenMMPlatform = OpenMMPlatform.Fastest,
):
    """
    Run small scale docking on a set of ligands, against a set of targets.
    """

    if input_json is not None:
        print("Loading inputs from json file... Will override all other inputs.")
        inputs = SmallScaleDockingInputs.from_json_file(input_json)

    else:
        inputs = SmallScaleDockingInputs(
            postera=postera,
            postera_upload=postera_upload,
            target=target,
            use_dask=use_dask,
            dask_type=dask_type,
            posit_confidence_cutoff=posit_confidence_cutoff,
            filename=ligands,
            pdb_file=pdb_file,
            fragalysis_dir=fragalysis_dir,
            structure_dir=structure_dir,
            postera_molset_name=postera_molset_name,
            cache_dir=cache_dir,
            gen_cache=gen_cache,
            cache_type=cache_type,
            ml_scorers=ml_scorer,
            output_dir=output_dir,
            md=md,
            md_steps=md_steps,
            md_openmm_platform=md_openmm_platform,
        )

    small_scale_docking_workflow(inputs)


if __name__ == "__main__":
    docking()
