import click
from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags
from asapdiscovery.data.util.dask_utils import DaskFailureMode, DaskType
from asapdiscovery.ml.models import ASAPMLModelRegistry
from asapdiscovery.simulation.simulate import OpenMMPlatform


def postera(func):
    return click.option(
        "--postera",
        is_flag=True,
        default=False,
        help="Whether to download complexes from Postera.",
    )(func)


def postera_molset_name(func):
    return click.option(
        "--postera-molset-name",
        type=str,
        default=None,
        help="The name of the Postera molecule set to use.",
    )(func)


def postera_upload(func):
    return click.option(
        "--postera-upload",
        is_flag=True,
        default=False,
        help="Whether to upload results to Postera.",
    )(func)


def postera_args(func):
    return postera(postera_molset_name(postera_upload(func)))


def use_dask(func):
    return click.option(
        "--use-dask",
        is_flag=True,
        default=False,
        help="Whether to use dask for parallelism.",
    )(func)


def dask_type(func):
    return click.option(
        "--dask-type",
        type=click.Choice(DaskType.get_values(), case_sensitive=False),
        default=DaskType.LOCAL,
        help="The type of dask cluster to use. Can be 'local', 'lilac-cpu' or  'lilac-gpu'.",
    )(func)


def dask_failure_mode(func):
    return click.option(
        "--dask-failure-mode",
        type=click.Choice(DaskFailureMode.get_values(), case_sensitive=False),
        default=DaskFailureMode.SKIP,
        help="The failure mode for dask. Can be 'raise' or 'skip'.",
        show_default=True,
    )(func)


def dask_args(func):
    return use_dask(dask_type(dask_failure_mode(func)))


def target(func):
    return click.option(
        "--target",
        type=click.Choice(TargetTags.get_values(), case_sensitive=True),
        help="The target for the workflow",
        required=True,
    )(func)


def ligands(func):
    return click.option(
        "-l",
        "--ligands",
        type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
        help="File containing ligands",
    )(func)


def output_dir(func):
    return click.option(
        "--output-dir",
        type=click.Path(
            resolve_path=True, exists=False, file_okay=False, dir_okay=True
        ),
        help="The directory to output results to.",
        default="output",
    )(func)


def overwrite(func):
    return click.option(
        "--overwrite/--no-overwrite",
        default=True,
        help="Whether to overwrite the output directory if it exists.",
    )(func)


def input_json(func):
    return click.option(
        "--input-json",
        type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
        help="Path to a json file containing the inputs to the workflow,  WARNING: overrides all other inputs.",
    )(func)


def ml_scorer(func):
    return click.option(
        "--ml-scorer",
        type=click.Choice(
            ASAPMLModelRegistry.get_implemented_model_types(), case_sensitive=True
        ),
        multiple=True,
        help="The names of the ml scorer to use, can be specified multiple times to use multiple ml scorers.",
    )(func)


def ml_scorer_bymodel(func):
    return click.option(
        "--ml-scorer-bymodel",
        type=str,
        multiple=True,
        help="Exact name of the ml model to use, can be specified multiple times to use multiple ml scorers.",
    )(func)


def ml_scorer_auto(func):
    return click.option(
        "--ml-scorer-auto/--no-ml-scorer-auto",
        help="Whether to automatically use latest available ml scorers for each model type available for target.",
        default=True,
    )(func)


def fragalysis_dir(func):
    return click.option(
        "--fragalysis-dir",
        type=click.Path(resolve_path=True, exists=True, file_okay=False, dir_okay=True),
        help="Path to a directory containing fragments to dock.",
    )(func)


def structure_dir(func):
    return click.option(
        "--structure-dir",
        type=click.Path(resolve_path=True, exists=True, file_okay=False, dir_okay=True),
        help="Path to a directory containing structures.",
    )(func)


def pdb_file(func):
    return click.option(
        "--pdb-file",
        type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
        help="Path to a pdb file containing a structure",
    )(func)


def cache_dir(func):
    return click.option(
        "--cache-dir",
        type=click.Path(
            resolve_path=True, exists=False, file_okay=False, dir_okay=True
        ),
        help="Path to a directory where design units are cached.",
    )(func)


def use_only_cache(func):
    return click.option(
        "--use-only-cache",
        is_flag=True,
        default=False,
        help="Whether to only use the cache.",
    )(func)


def gen_cache_w_default(func):
    return click.option(
        "--gen-cache",
        type=click.Path(
            resolve_path=False, exists=False, file_okay=False, dir_okay=True
        ),
        help="Path to a directory where a design unit cache should be generated.",
        default="prepped_structure_cache",
    )(func)


def md(func):
    return click.option(
        "--md",
        is_flag=True,
        default=False,
        help="Whether to run MD",
    )(func)


def md_steps(func):
    return click.option(
        "--md-steps",
        type=int,
        default=2500000,
        help="Number of MD steps",
    )(func)


def md_openmm_platform(func):
    return click.option(
        "--md-openmm-platform",
        type=click.Choice(OpenMMPlatform.get_values(), case_sensitive=False),
        default=OpenMMPlatform.Fastest,
        help="The OpenMM platform to use for MD",
    )(func)


def md_args(func):
    return md(md_steps(md_openmm_platform(func)))


def core_smarts(func):
    return click.option(
        "-cs",
        "--core-smarts",
        type=click.STRING,
        help="The SMARTS which should be used to select which atoms to constrain to the reference structure.",
    )(func)


def save_to_cache(func):
    return click.option(
        "--save-to-cache/--no-save-to-cache",
        help="If the newly generated structures should be saved to the cache folder.",
        default=True,
    )(func)


def loglevel(func):
    return click.option(
        "--loglevel",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
        help="The log level to use.",
        default="INFO",
        show_default=True,
    )(func)


def walltime(func):
    return click.option(
        "--walltime",
        type=str,
        default="72h",
        help="The walltime to use for the dask_jobqueue cluster (if used) in dask format, e.g. '1h'",
    )(func)
