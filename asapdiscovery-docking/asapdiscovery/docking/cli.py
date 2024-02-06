from typing import Optional

import click
from asapdiscovery.cli.cli_args import (
    cache_dir,
    dask_args,
    fragalysis_dir,
    input_json,
    ligands,
    md_args,
    ml_scorer,
    output_dir,
    overwrite,
    pdb_file,
    postera_args,
    save_to_cache,
    structure_dir,
    target,
    use_only_cache,
    walltime,
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
from asapdiscovery.docking.workflows.small_scale_docking import (
    SmallScaleDockingInputs,
    small_scale_docking_workflow,
)
from asapdiscovery.simulation.simulate import OpenMMPlatform


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
@save_to_cache
@cache_dir
@dask_args
@output_dir
@overwrite
@input_json
@ml_scorer
@walltime
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
    save_to_cache: Optional[bool] = True,
    cache_dir: Optional[str] = None,
    output_dir: str = "output",
    overwrite: bool = True,
    input_json: Optional[str] = None,
    use_dask: bool = False,
    dask_type: DaskType = DaskType.LOCAL,
    ml_scorer: Optional[list[str]] = None,
    walltime: Optional[str] = "72h",
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
            ligands=ligands,
            pdb_file=pdb_file,
            fragalysis_dir=fragalysis_dir,
            structure_dir=structure_dir,
            postera_molset_name=postera_molset_name,
            cache_dir=cache_dir,
            save_to_cache=save_to_cache,
            ml_scorers=ml_scorer,
            output_dir=output_dir,
            overwrite=overwrite,
            walltime=walltime,
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
    "--omega-dense",
    is_flag=True,
    default=False,
    help="Whether to use dense conformer enumeration with OEOmega (slower, more accurate)",
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
    type=click.Choice(StructureSelector.get_values(), case_sensitive=False),
    default=StructureSelector.LEAVE_SIMILAR_OUT,
    help="The type of structure selector to use.",
)
@ligands
@pdb_file
@fragalysis_dir
@structure_dir
@save_to_cache
@cache_dir
@use_only_cache
@dask_args
@output_dir
@overwrite
@input_json
@walltime
def cross_docking(
    target: TargetTags,
    multi_reference: bool = False,
    structure_selector: StructureSelector = StructureSelector.LEAVE_SIMILAR_OUT,
    use_omega: bool = False,
    omega_dense: bool = False,
    allow_retries: bool = False,
    allow_final_clash: bool = False,
    ligands: Optional[str] = None,
    pdb_file: Optional[str] = None,
    fragalysis_dir: Optional[str] = None,
    structure_dir: Optional[str] = None,
    use_only_cache: bool = False,
    save_to_cache: Optional[bool] = True,
    cache_dir: Optional[str] = None,
    output_dir: str = "output",
    overwrite: bool = True,
    input_json: Optional[str] = None,
    use_dask: bool = False,
    dask_type: DaskType = DaskType.LOCAL,
    walltime: Optional[str] = "72h",
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
            omega_dense=omega_dense,
            allow_retries=allow_retries,
            ligands=ligands,
            pdb_file=pdb_file,
            fragalysis_dir=fragalysis_dir,
            structure_dir=structure_dir,
            cache_dir=cache_dir,
            use_only_cache=use_only_cache,
            save_to_cache=save_to_cache,
            output_dir=output_dir,
            overwrite=overwrite,
            allow_final_clash=allow_final_clash,
            walltime=walltime,
        )

    cross_docking_workflow(inputs)


@docking.command()
@target
@click.option(
    "--posit-confidence-cutoff",
    type=float,
    default=0.1,
    help="The confidence cutoff for POSIT results to be considered",
)
@click.option("--allow-dask-cuda/--no-allow-dask-cuda", default=True)
@click.option(
    "--no-omega",
    is_flag=True,
    default=False,
    help="Whether to use OEOmega conformer enumeration before docking (slower, more accurate)",
)
@ligands
@postera_args
@pdb_file
@fragalysis_dir
@structure_dir
@save_to_cache
@cache_dir
@dask_args
@output_dir
@overwrite
@input_json
@ml_scorer
@md_args
@walltime
def small_scale(
    target: TargetTags,
    posit_confidence_cutoff: float = 0.1,
    allow_dask_cuda: bool = True,
    no_omega: bool = False,
    ligands: Optional[str] = None,
    postera: bool = False,
    postera_molset_name: Optional[str] = None,
    postera_upload: bool = False,
    pdb_file: Optional[str] = None,
    fragalysis_dir: Optional[str] = None,
    structure_dir: Optional[str] = None,
    save_to_cache: Optional[bool] = True,
    cache_dir: Optional[str] = None,
    output_dir: str = "output",
    overwrite: bool = True,
    input_json: Optional[str] = None,
    use_dask: bool = False,
    dask_type: DaskType = DaskType.LOCAL,
    ml_scorer: Optional[list[str]] = None,
    md: bool = False,
    md_steps: int = 2500000,  # 10 ns @ 4.0 fs timestep
    md_openmm_platform: OpenMMPlatform = OpenMMPlatform.Fastest,
    walltime: Optional[str] = "72h",
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
            allow_dask_cuda=allow_dask_cuda,
            use_omega=not no_omega,
            ligands=ligands,
            pdb_file=pdb_file,
            fragalysis_dir=fragalysis_dir,
            structure_dir=structure_dir,
            postera_molset_name=postera_molset_name,
            cache_dir=cache_dir,
            save_to_cache=save_to_cache,
            ml_scorers=ml_scorer,
            output_dir=output_dir,
            overwrite=overwrite,
            md=md,
            md_steps=md_steps,
            md_openmm_platform=md_openmm_platform,
            walltime=walltime,
        )

    small_scale_docking_workflow(inputs)


if __name__ == "__main__":
    docking()
