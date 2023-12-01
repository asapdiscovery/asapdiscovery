import json
from pathlib import Path

import click
from asapdiscovery.ml.cli_args import (
    output_dir,
    config_file,
    use_wandb,
    sweep,
    proj,
    name,
    extra_config,
    grouped,
    strategy,
    pred_readout,
    combination,
    comb_readout,
    max_comb_neg,
    max_comb_scale,
    pred_substrate,
    pred_km,
    comb_substrate,
    comb_km,
    in_feats,
    num_layers_gat,
    hidden_feats,
    num_heads,
    feat_drops,
    attn_drops,
    alphas,
    residuals,
    agg_modes,
    biases,
    allow_zero_in_degree,
    hidden_channels,
    num_filters,
    num_interactions,
    num_gaussians,
    cutoff,
    max_num_neighbors,
    readout,
    dipole,
    mean,
    std,
    num_atom_types,
    irreps_hidden,
    lig,
    irreps_edge_attr,
    num_layers_schnet,
    neighbor_dist,
    num_basis,
    num_radial_layers,
    num_radial_neurons,
    num_neighbors,
    num_nodes,
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


@build_and_train.command()
def gat():
    print("gat", flush=True)


@build_and_train.command()
def schnet():
    print("schnet", flush=True)


@build_and_train.command()
def e3nn():
    print("e3nn", flush=True)


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
@num_layers_schnet
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
    num_layers_schnet: int | None = None,
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
                "num_layers": num_layers_schnet,
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
