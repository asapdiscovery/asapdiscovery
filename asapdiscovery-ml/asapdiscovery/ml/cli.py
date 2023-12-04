import json
from pathlib import Path

import click
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.utils import (
    MOONSHOT_CDD_ID_REGEX,
    MPRO_ID_REGEX,
    extract_compounds_from_filenames,
)
from asapdiscovery.ml.cli_args import (
    agg_modes,
    allow_zero_in_degree,
    alphas,
    attn_drops,
    biases,
    comb_km,
    comb_readout,
    comb_substrate,
    combination,
    config_file,
    cutoff,
    dipole,
    ds_cache,
    ds_config_cache,
    e3nn_args,
    exp_file,
    extra_config,
    feat_drops,
    gat_args,
    grouped,
    hidden_channels,
    hidden_feats,
    in_feats,
    irreps_edge_attr,
    irreps_hidden,
    lig,
    max_comb_neg,
    max_comb_scale,
    max_num_neighbors,
    mean,
    mtenn_args,
    name,
    neighbor_dist,
    num_atom_types,
    num_basis,
    num_filters,
    num_gaussians,
    num_heads,
    num_interactions,
    num_layers_e3nn,
    num_layers_gat,
    num_neighbors,
    num_nodes,
    num_radial_layers,
    num_radial_neurons,
    output_dir,
    pred_km,
    pred_readout,
    pred_substrate,
    proj,
    readout,
    residuals,
    schnet_args,
    std,
    strategy,
    str_files,
    str_fn_cpd_regex,
    str_fn_xtal_regex,
    sweep,
    use_wandb,
    wandb_args,
)
from asapdiscovery.ml.schema_v2.config import (
    DatasetConfig,
    DatasetType,
)
from mtenn.config import (
    CombinationConfig,
    E3NNModelConfig,
    GATModelConfig,
    ModelType,
    ReadoutConfig,
    SchNetModelConfig,
    StrategyConfig,
)


@click.group()
def cli():
    pass


# Functions for just building a Trainer and then dumping it
@click.group()
def build():
    pass


# Function for training using an already built Trainer
@cli.command()
def train():
    pass


# Functions for building a Trainer and subsequently training the model
@click.group(name="build-and-train")
def build_and_train():
    pass


cli.add_command(build)
cli.add_command(build_and_train)


@build.command(name="gat")
def build_gat():
    pass


@build.command(name="schnet")
def build_schnet():
    pass


@build.command(name="e3nn")
def build_e3nn():
    pass


@build_and_train.command(name="gat")
@output_dir
@exp_file
@ds_cache
@ds_config_cache
@config_file
@wandb_args
@mtenn_args
@gat_args
def build_and_train_gat(
    output_dir: Path,
    exp_file: Path | None = None,
    ds_cache: Path | None = None,
    ds_config_cache: Path | None = None,
    config_file: Path | None = None,
    use_wandb: bool = False,
    sweep: bool = False,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    extra_config: list[str] | None = None,
    grouped: bool | None = None,
    strategy: StrategyConfig | None = None,
    pred_readout: ReadoutConfig | None = None,
    combination: CombinationConfig | None = None,
    comb_readout: ReadoutConfig | None = None,
    max_comb_neg: bool | None = None,
    max_comb_scale: float | None = None,
    pred_substrate: float | None = None,
    pred_km: float | None = None,
    comb_substrate: float | None = None,
    comb_km: float | None = None,
    in_feats: int | None = None,
    num_layers_gat: int | None = None,
    hidden_feats: str | None = None,
    num_heads: str | None = None,
    feat_drops: str | None = None,
    attn_drops: str | None = None,
    alphas: str | None = None,
    residuals: str | None = None,
    agg_modes: str | None = None,
    biases: str | None = None,
    allow_zero_in_degree: bool | None = None,
):
    ds_config = _build_ds_config(
        exp_file=exp_file,
        structures=None,
        xtal_regex=None,
        cpd_regex=None,
        ds_cache=ds_cache,
        ds_config_cache=ds_config_cache,
        is_structural=False,
        is_grouped=grouped,
    )
    pass


@build_and_train.command()
@output_dir
@exp_file
@str_files
@str_fn_cpd_regex
@str_fn_xtal_regex
@ds_cache
@ds_config_cache
@config_file
@wandb_args
@mtenn_args
@schnet_args
def build_and_train_schnet(
    output_dir: Path,
    exp_file: Path | None = None,
    structures: Path | None = None,
    cpd_regex: str = MOONSHOT_CDD_ID_REGEX,
    xtal_regex: str = MPRO_ID_REGEX,
    ds_cache: Path | None = None,
    ds_config_cache: Path | None = None,
    config_file: Path | None = None,
    use_wandb: bool = False,
    sweep: bool = False,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    extra_config: list[str] | None = None,
    grouped: bool | None = None,
    strategy: StrategyConfig | None = None,
    pred_readout: ReadoutConfig | None = None,
    combination: CombinationConfig | None = None,
    comb_readout: ReadoutConfig | None = None,
    max_comb_neg: bool | None = None,
    max_comb_scale: float | None = None,
    pred_substrate: float | None = None,
    pred_km: float | None = None,
    comb_substrate: float | None = None,
    comb_km: float | None = None,
    hidden_channels: int | None = None,
    num_filters: int | None = None,
    num_interactions: int | None = None,
    num_gaussians: int | None = None,
    cutoff: float | None = None,
    max_num_neighbors: int | None = None,
    readout: str | None = None,
    dipole: bool | None = None,
    mean: float | None = None,
    std: float | None = None,
):
    pass


@build_and_train.command()
@output_dir
@exp_file
@str_files
@str_fn_cpd_regex
@str_fn_xtal_regex
@ds_cache
@ds_config_cache
@config_file
@wandb_args
@mtenn_args
@e3nn_args
def build_and_train_e3nn(
    output_dir: Path,
    exp_file: Path | None = None,
    structures: Path | None = None,
    cpd_regex: str = MOONSHOT_CDD_ID_REGEX,
    xtal_regex: str = MPRO_ID_REGEX,
    ds_cache: Path | None = None,
    ds_config_cache: Path | None = None,
    config_file: Path | None = None,
    use_wandb: bool = False,
    sweep: bool = False,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    extra_config: list[str] | None = None,
    grouped: bool | None = None,
    strategy: StrategyConfig | None = None,
    pred_readout: ReadoutConfig | None = None,
    combination: CombinationConfig | None = None,
    comb_readout: ReadoutConfig | None = None,
    max_comb_neg: bool | None = None,
    max_comb_scale: float | None = None,
    pred_substrate: float | None = None,
    pred_km: float | None = None,
    comb_substrate: float | None = None,
    comb_km: float | None = None,
    num_atom_types: int | None = None,
    irreps_hidden: str | None = None,
    lig: bool | None = None,
    irreps_edge_attr: int | None = None,
    num_layers_e3nn: int | None = None,
    neighbor_dist: float | None = None,
    num_basis: int | None = None,
    num_radial_layers: int | None = None,
    num_radial_neurons: int | None = None,
    num_neighbors: float | None = None,
    num_nodes: float | None = None,
):
    pass


@cli.command()
@output_dir
# Model setup args
@config_file
# W&B args
@use_wandb
@sweep
@proj
@name
@extra_config
# Shared MTENN-related parameters
@grouped
@strategy
@pred_readout
@combination
@comb_readout
@max_comb_neg
@max_comb_scale
@pred_substrate
@pred_km
@comb_substrate
@comb_km
# GAT-specific parameters
@in_feats
@num_layers_gat
@hidden_feats
@num_heads
@feat_drops
@attn_drops
@alphas
@residuals
@agg_modes
@biases
@allow_zero_in_degree
# SchNet-specific parameters
@hidden_channels
@num_filters
@num_interactions
@num_gaussians
@cutoff
@max_num_neighbors
@readout
@dipole
@mean
@std
# e3nn-specific parameters
@num_atom_types
@irreps_hidden
@lig
@irreps_edge_attr
@num_layers_e3nn
@neighbor_dist
@num_basis
@num_radial_layers
@num_radial_neurons
@num_neighbors
@num_nodes
def test(
    output_dir: Path,
    model_type: ModelType,
    config_file: Path | None = None,
    use_wandb: bool = False,
    sweep: bool = False,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    extra_config: list[str] | None = None,
    grouped: bool | None = None,
    strategy: StrategyConfig | None = None,
    pred_readout: ReadoutConfig | None = None,
    combination: CombinationConfig | None = None,
    comb_readout: ReadoutConfig | None = None,
    max_comb_neg: bool | None = None,
    max_comb_scale: float | None = None,
    pred_substrate: float | None = None,
    pred_km: float | None = None,
    comb_substrate: float | None = None,
    comb_km: float | None = None,
    in_feats: int | None = None,
    num_layers_gat: int | None = None,
    hidden_feats: str | None = None,
    num_heads: str | None = None,
    feat_drops: str | None = None,
    attn_drops: str | None = None,
    alphas: str | None = None,
    residuals: str | None = None,
    agg_modes: str | None = None,
    biases: str | None = None,
    allow_zero_in_degree: bool | None = None,
    hidden_channels: int | None = None,
    num_filters: int | None = None,
    num_interactions: int | None = None,
    num_gaussians: int | None = None,
    cutoff: float | None = None,
    max_num_neighbors: int | None = None,
    readout: str | None = None,
    dipole: bool | None = None,
    mean: float | None = None,
    std: float | None = None,
    num_atom_types: int | None = None,
    irreps_hidden: str | None = None,
    lig: bool | None = None,
    irreps_edge_attr: int | None = None,
    num_layers_e3nn: int | None = None,
    neighbor_dist: float | None = None,
    num_basis: int | None = None,
    num_radial_layers: int | None = None,
    num_radial_neurons: int | None = None,
    num_neighbors: float | None = None,
    num_nodes: float | None = None,
):
    # Build the model
    match model_type:
        case ModelType.GAT:
            config_class = GATModelConfig
            cli_config_vals = {
                "grouped": grouped,
                "strategy": strategy,
                "pred_readout": pred_readout,
                "combination": combination,
                "comb_readout": comb_readout,
                "max_comb_neg": max_comb_neg,
                "max_comb_scale": max_comb_scale,
                "pred_substrate": pred_substrate,
                "pred_km": pred_km,
                "comb_substrate": comb_substrate,
                "comb_km": comb_km,
                "in_feats": in_feats,
                "num_layers": num_layers_gat,
                "hidden_feats": hidden_feats,
                "num_heads": num_heads,
                "feat_drops": feat_drops,
                "attn_drops": attn_drops,
                "alphas": alphas,
                "residuals": residuals,
                "agg_modes": agg_modes,
                "biases": biases,
                "allow_zero_in_degree": allow_zero_in_degree,
            }
        case ModelType.schnet:
            config_class = SchNetModelConfig
            cli_config_vals = {
                "grouped": grouped,
                "strategy": strategy,
                "pred_readout": pred_readout,
                "combination": combination,
                "comb_readout": comb_readout,
                "max_comb_neg": max_comb_neg,
                "max_comb_scale": max_comb_scale,
                "pred_substrate": pred_substrate,
                "pred_km": pred_km,
                "comb_substrate": comb_substrate,
                "comb_km": comb_km,
                "hidden_channels": hidden_channels,
                "num_filters": num_filters,
                "num_interactions": num_interactions,
                "num_gaussians": num_gaussians,
                "cutoff": cutoff,
                "max_num_neighbors": max_num_neighbors,
                "readout": readout,
                "dipole": dipole,
                "mean": mean,
                "std": std,
            }
        case ModelType.e3nn:
            config_class = E3NNModelConfig
            cli_config_vals = {
                "grouped": grouped,
                "strategy": strategy,
                "pred_readout": pred_readout,
                "combination": combination,
                "comb_readout": comb_readout,
                "max_comb_neg": max_comb_neg,
                "max_comb_scale": max_comb_scale,
                "pred_substrate": pred_substrate,
                "pred_km": pred_km,
                "comb_substrate": comb_substrate,
                "comb_km": comb_km,
                "num_atom_types": num_atom_types,
                "irreps_hidden": irreps_hidden,
                "lig": lig,
                "irreps_edge_attr": irreps_edge_attr,
                "num_layers": num_layers_e3nn,
                "neighbor_dist": neighbor_dist,
                "num_basis": num_basis,
                "num_radial_layers": num_radial_layers,
                "num_radial_neurons": num_radial_neurons,
                "num_neighbors": num_neighbors,
                "num_nodes": num_nodes,
            }
        case unknown:
            raise ValueError(f"Unknown model type: {unknown}")

    # Only keep values CLI config vals that actually had a value passed
    cli_config_vals = {k: v for k, v in cli_config_vals.items() if v is not None}
    print(cli_config_vals, flush=True)

    # Parse config file (if given), and reconcile those args with CLI args
    if config_file:
        fn_config_vals = json.load(config_file.open())
    else:
        fn_config_vals = {}
    # Want CLI args to overwrite file args
    config_vals = fn_config_vals | cli_config_vals

    config = config_class(**config_vals)

    print(config, flush=True)
    model = config.build()
    print(model, flush=True)


def _check_ds_args(exp_file, structures, ds_cache, ds_config_cache, is_structural):
    """
    Helper function to check that all necessary dataset files were passed.

    Parameters
    ----------
    exp_file : Path
        JSON file giving a list of ExperimentalDataCompound objects
    structures : Path
        Glob or directory containing PDB files
    ds_cache : Path
        Dataset cache file
    ds_config_cache : Path
        Dataset config cache function

    Returns
    -------
    bool
        Whether an appropriate combination of args was passed
    """
    # Can just load from the config cache file so don't need anything else
    if ds_config_cache and ds_config_cache.exists():
        return True

    # Otherwise need to load data so make sure they all exist
    if (not exp_file) or (not exp_file.exists()):
        return False
    if is_structural:
        if not structures:
            return False
        if structures.is_dir():
            # Make sure there's at least one PDB file
            try:
                _ = next(iter(structures.glob("*.pdb")))
            except StopIteration:
                return False
        else:
            # Make sure there's at least one file that matches the glob
            try:
                _ = next(iter(structures.parent.glob(structures.name)))
            except StopIteration:
                return False

    # Nothing has failed so we should be good to go
    return True


def _build_ds_config(
    exp_file,
    structures,
    xtal_regex,
    cpd_regex,
    ds_cache,
    ds_config_cache,
    is_structural,
    is_grouped,
):
    """
    Helper function to build a DatasetConfig object.

    Parameters
    ----------
    exp_file : Path
        JSON file giving a list of ExperimentalDataCompound objects
    structures : Path
        Glob or directory containing PDB files
    ds_cache : Path
        Dataset cache file
    ds_config_cache : Path
        Dataset config cache function

    Returns
    -------
    DatasetConfig
        DatasetConfig object
    """

    if not _check_ds_args(
        exp_file=exp_file,
        structures=None,
        ds_cache=ds_cache,
        ds_config_cache=ds_config_cache,
        is_structural=False,
    ):
        raise ValueError("Invalid combination of dataset args.")

    if ds_config_cache and ds_config_cache.exists():
        return json.loads(ds_config_cache.read_text())

    # Pick correct DatasetType
    if is_structural:
        ds_type = DatasetType.structural
    else:
        ds_type = DatasetType.graph

    # Parse experimental data
    exp_compounds = json.loads(exp_file.read_text())
    exp_data = {
        c.compound_id: c.experimental_data | {"date_created": c.date_created}
        for c in exp_compounds
    }

    # Create Ligand/Complex objects
    if is_structural:
        if structures.is_dir():
            all_str_fns = structures.glob("*.pdb")
        else:
            all_str_fns = structures.parent.glob(structures.name)
        compounds = extract_compounds_from_filenames(
            all_str_fns, xtal_pat=xtal_regex, compound_pat=cpd_regex, fail_val="NA"
        )
        input_data = [
            Complex.from_pdb(
                fn,
                target_kwargs={"target_name": cpd[0]},
                ligand_kwargs={"compound_name": cpd[1]},
            )
            for fn, cpd in zip(all_str_fns, compounds)
        ]
    else:
        input_data = [
            Ligand.from_smiles(c.smiles, compound_name=c.compound_id)
            for c in exp_compounds
        ]

    ds_config = DatasetConfig(
        ds_type=ds_type,
        exp_data=exp_data,
        input_data=input_data,
        cache_file=ds_cache,
        grouped=is_grouped,
    )

    # Save file if desired
    if ds_config_cache:
        ds_config_cache.write_text(ds_config.json())

    return ds_config
