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
)
from asapdiscovery.data.dask_utils import DaskType
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.data.selectors.selector_list import StructureSelector
from asapdiscovery.docking.workflows.cross_docking import (
    CrossDockingWorkflowInputs,
    cross_docking_workflow,
)
from asapdiscovery.docking.workflows.large_scale_docking import (
    LargeScaleDockingInputs,
    large_scale_docking_workflow,
)


@click.group()
def docking():
    pass


@docking.command()
@target
@click.option(
    "--n-select",
    type=int,
    default=5,
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
@click.option(
    "--use-omega",
    is_flag=True,
    default=False,
    help="Whether to use OEOmega conformer enumeration before docking (slower, more accurate)",
)
@click.option(
    "--allow-posit-retries",
    is_flag=True,
    default=False,
    help="Whether to allow POSIT to retry with relaxed parameters if docking fails (slower, more likely to succeed)",
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
    use_omega: bool = False,
    allow_posit_retries: bool = False,
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
            use_omega=use_omega,
            allow_posit_retries=allow_posit_retries,
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
    "--use-omega",
    is_flag=True,
    default=False,
    help="Whether to use OEOmega conformer enumeration before docking (slower, more accurate)",
)
@click.option(
    "--allow-retries",
    is_flag=True,
    default=False,
    help="Whether to allow POSIT to retry with relaxed parameters if docking fails (slower, more likely to succeed)",
)
@click.option(
    "--allow-final-clash",
    is_flag=True,
    default=False,
    help="Allow clashing poses in last stage of docking",
)
@click.option(
    "--multi-reference",
    is_flag=True,
    default=False,
    help="Whether to pass multiple references to the docker for each ligand instead of just one at a time",
)
@click.option(
    "--structure-selector",
    type=click.Choice(
        [selector.name for selector in StructureSelector], case_sensitive=True
    ),
    default=StructureSelector.PAIRWISE,
    help="The type of structure selector to use. Defaults to pairwise (all pairwise combinations of ligand and complex)",
)
@ligands
@pdb_file
@fragalysis_dir
@structure_dir
@gen_cache
@cache_dir
@cache_type
@dask_args
@output_dir
@input_json
def cross_docking(
    target: TargetTags,
    multi_reference: bool = False,
    structure_selector: StructureSelector = StructureSelector.PAIRWISE,
    use_omega: bool = False,
    allow_retries: bool = False,
    allow_final_clash: bool = False,
    ligands: Optional[str] = None,
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
):
    """
    Run cross docking on a set of ligands, against a set of targets.
    """

    if input_json is not None:
        print("Loading inputs from json file... Will override all other inputs.")
        inputs = CrossDockingWorkflowInputs.from_json_file(input_json)

    else:
        inputs = CrossDockingWorkflowInputs(
            target=target,
            multi_reference=multi_reference,
            structure_selector=structure_selector,
            use_dask=use_dask,
            dask_type=dask_type,
            use_omega=use_omega,
            allow_retries=allow_retries,
            filename=ligands,
            pdb_file=pdb_file,
            fragalysis_dir=fragalysis_dir,
            structure_dir=structure_dir,
            cache_dir=cache_dir,
            gen_cache=gen_cache,
            cache_type=cache_type,
            output_dir=output_dir,
            allow_final_clash=allow_final_clash,
        )

    cross_docking_workflow(inputs)


if __name__ == "__main__":
    docking()
