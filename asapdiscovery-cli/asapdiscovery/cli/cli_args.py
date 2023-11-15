import click
from asapdiscovery.data.dask_utils import DaskType
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.ml.models.ml_models import ASAPMLModelRegistry
from asapdiscovery.modeling.protein_prep_v2 import CacheType


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


def dask_args(func):
    return use_dask(dask_type(func))


def target(func):
    return click.option(
        "--target",
        type=click.Choice(TargetTags.get_values(), case_sensitive=True),
        help="The target for the workflow",
        required=True,
    )(func)


def ligands(func):
    return click.option(
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
        type=click.Path(resolve_path=True, exists=True, file_okay=False, dir_okay=True),
        help="Path to a directory where design units are cached.",
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


def gen_cache(func):
    return click.option(
        "--gen-cache",
        type=click.Path(
            resolve_path=False, exists=False, file_okay=False, dir_okay=True
        ),
        help="Path to a directory where a design unit cache should be generated.",
    )(func)


def cache_type(func):
    return click.option(
        "--cache-type",
        type=click.Choice(CacheType.get_values(), case_sensitive=False),
        default=[CacheType.DesignUnit],
        multiple=True,
        help="The type of cache to use, can be 'JSON' or 'DesignUnit', an be specified multiple times to use cache",
    )(func)
