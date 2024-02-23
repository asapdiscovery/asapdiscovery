import json
from glob import glob
from pathlib import Path

import click
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.util.utils import (
    MOONSHOT_CDD_ID_REGEX,
    MPRO_ID_REGEX,
    extract_compounds_from_filenames,
)
from asapdiscovery.ml.cli_args import (
    ds_cache,
    grouped,
    str_files,
    str_fn_cpd_regex,
    str_fn_xtal_regex,
)
from asapdiscovery.ml.config import DatasetConfig, DatasetType


@click.command()
@click.option(
    "-from",
    "--convert-from",
    required=True,
    type=DatasetType,
    help=(
        "DatasetType to convert from. Options are "
        f"[{', '.join(DatasetType.get_values())}]."
    ),
)
@click.option(
    "-to",
    "--convert-to",
    required=True,
    type=DatasetType,
    help=(
        "DatasetType to convert to. Options are "
        f"[{', '.join(DatasetType.get_values())}]."
    ),
)
@click.option(
    "-in",
    "--in-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="DatasetConfig JSON file to load.",
)
@click.option(
    "-out",
    "--out-file",
    required=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="DatasetConfig JSON file to export.",
)
@grouped
@str_files
@str_fn_xtal_regex
@str_fn_cpd_regex
@click.option(
    "--e3nn",
    is_flag=True,
    help=(
        "If converting to a structural dataset, whether the output will be used with "
        "an e3nn model."
    ),
)
@ds_cache
def convert(
    convert_from: DatasetType,
    convert_to: DatasetType,
    in_file: Path,
    out_file: Path,
    grouped: bool | None = None,
    structures: str | None = None,
    xtal_regex: str = MPRO_ID_REGEX,
    cpd_regex: str = MOONSHOT_CDD_ID_REGEX,
    e3nn: bool = False,
    ds_cache: Path | None = None,
):
    if convert_from == DatasetType.graph and convert_to == DatasetType.structural:
        convert_graph_to_struct(in_file, out_file, structures, e3nn, grouped, ds_cache)
    elif convert_from == DatasetType.structural and convert_to == DatasetType.graph:
        convert_struct_to_graph(in_file, out_file)
    else:
        print("Input and output types are the same, not doing anything.")


def convert_graph_to_struct(
    in_file: Path,
    out_file: Path,
    structures: str | None = None,
    xtal_regex: str = MPRO_ID_REGEX,
    cpd_regex: str = MOONSHOT_CDD_ID_REGEX,
    e3nn: bool = False,
    grouped: bool | None = None,
    ds_cache: Path | None = None,
):
    in_config = DatasetConfig(**json.loads(in_file.read_text()))
    if Path(structures).is_dir():
        all_str_fns = Path(structures).glob("*.pdb")
    else:
        all_str_fns = glob(structures)
    compounds = extract_compounds_from_filenames(
        all_str_fns, xtal_pat=xtal_regex, compound_pat=cpd_regex, fail_val="NA"
    )

    # Filter compounds to only include datat that we have experimental data for
    idx = [c[1] in in_config.exp_data for c in compounds]
    print(
        f"Filtering {len(idx) - sum(idx)} structures that we don't have",
        "experimental data for.",
        flush=True,
    )
    compounds = [c for i, c in zip(idx, compounds) if i]
    all_str_fns = [fn for i, fn in zip(idx, all_str_fns) if i]

    print(len(all_str_fns), len(compounds), flush=True)
    input_data = [
        Complex.from_pdb(
            fn,
            target_kwargs={"target_name": cpd[0]},
            ligand_kwargs={"compound_name": cpd[1]},
        )
        for fn, cpd in zip(all_str_fns, compounds)
    ]

    config_kwargs = {
        "ds_type": DatasetType.structural,
        "exp_data": in_config.exp_data,
        "input_data": input_data,
        "cache_file": ds_cache,
        "for_e3nn": e3nn,
    }
    if grouped is not None:
        config_kwargs["grouped"] = grouped
    ds_config = DatasetConfig(**config_kwargs)
    out_file.write_text(ds_config.json())


def convert_struct_to_graph(
    in_file: Path,
    out_file: Path,
    ds_cache: Path | None = None,
):
    in_config = DatasetConfig(**json.loads(in_file.read_text()))
    input_data = [comp.ligand for comp in in_config.input_data]

    config_kwargs = {
        "ds_type": DatasetType.graph,
        "exp_data": in_config.exp_data,
        "input_data": input_data,
        "cache_file": ds_cache,
        "for_e3nn": False,
    }
    ds_config = DatasetConfig(**config_kwargs)
    out_file.write_text(ds_config.json())
