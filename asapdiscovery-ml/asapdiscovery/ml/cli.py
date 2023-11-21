import json
from pathlib import Path

import asapdiscovery.ml.schema_v2.config as ascfg
import click
from asapdiscovery.ml.models import MLModelType


@click.group()
def ml():
    pass


@ml.command()
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help=(
        "Top-level output directory. A subdirectory with the current W&B "
        "run ID will be made/searched if W&B is being used."
    ),
)
# Model setup args
@click.option(
    "-model",
    "--model-type",
    required=True,
    type=MLModelType,
    help="Which model type to use.",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "JSON file giving model config. Any passed CLI args will overwrite the options "
        "in this file."
    ),
)
# W&B args
@click.option("--use-wandb", is_flag=True, help="Use W&B to log model training.")
@click.option("--sweep", is_flag=True, help="This run is part of a W&B sweep.")
@click.option("-proj", "--wandb-project", help="W&B project name.")
@click.option("-name", "--wandb-name", help="W&B project name.")
@click.option(
    "-e",
    "--extra_config",
    multiple=True,
    help=(
        "Any extra config options to log to W&B, provided as comma-separated pairs. "
        "Can be provided as many times as desired "
        "(eg -e key1,val1 -e key2,val2 -e key3,val3)."
    ),
)
# Shared MTENN-related parameters
@click.option(
    "--grouped",
    type=bool,
    help="Model is a grouped (multi-pose) model.",
)
@click.option(
    "--strategy",
    type=ascfg.MTENNStrategy,
    help=(
        "Which Strategy to use for combining complex, protein, and ligand "
        "representations in the MTENN Model."
    ),
)
@click.option(
    "--pred-readout",
    type=ascfg.MTENNReadout,
    help=(
        "Which Readout to use for the model predictions. This corresponds "
        "to the individual pose predictions in the case of a GroupedModel."
    ),
)
@click.option(
    "--combination",
    type=ascfg.MTENNCombination,
    help="Which Combination to use for combining predictions in a GroupedModel.",
)
@click.option(
    "--comb-readout",
    type=ascfg.MTENNReadout,
    help=(
        "Which Readout to use for the combined model predictions. This is only "
        "relevant in the case of a GroupedModel."
    ),
)
@click.option(
    "--max-comb-neg",
    type=bool,
    help=(
        "Whether to take the min instead of max when combining pose predictions "
        "with MaxCombination."
    ),
)
@click.option(
    "--max-comb-scale",
    type=float,
    help=(
        "Scaling factor for values when taking the max/min when combining pose "
        "predictions with MaxCombination. A value of 1 will approximate the "
        "Boltzmann mean, while a larger value will more accurately approximate the "
        "max/min operation."
    ),
)
@click.option(
    "--pred-substrate",
    type=float,
    help=(
        "Substrate concentration to use when using the Cheng-Prusoff equation to "
        "convert deltaG -> IC50 in PIC50Readout for pred_readout. Assumed to be in "
        "the same units as pred_km."
    ),
)
@click.option(
    "--pred-km",
    type=float,
    help=(
        "Km value to use when using the Cheng-Prusoff equation to convert "
        "deltaG -> IC50 in PIC50Readout for pred_readout. Assumed to be in "
        "the same units as pred_substrate."
    ),
)
@click.option(
    "--comb-substrate",
    type=float,
    help=(
        "Substrate concentration to use when using the Cheng-Prusoff equation to "
        "convert deltaG -> IC50 in PIC50Readout for comb_readout. Assumed to be in "
        "the same units as comb_km."
    ),
)
@click.option(
    "--comb-km",
    type=float,
    help=(
        "Km value to use when using the Cheng-Prusoff equation to convert "
        "deltaG -> IC50 in PIC50Readout for comb_readout. Assumed to be in "
        "the same units as comb_substrate."
    ),
)
# GAT-specific parameters
@click.option("--in-feats", type=int, help="Input node feature size.")
@click.option(
    "--num-layers-gat",
    type=int,
    help=(
        "Number of GAT layers. Ignored if multiple values are passed for any "
        "other GAT argument. To define a model with only one layer, this must be "
        "explicitly set to 1."
    ),
)
@click.option(
    "--hidden-feats",
    help=(
        "Output size of each GAT layer. This can either be a single value, which will "
        "be broadcasted to each layer, or a comma-separated list with each value "
        "corresponding to one layer in the model."
    ),
)
@click.option(
    "--num-heads",
    help=(
        "Number of attention heads for each GAT layer. Passing a single value or "
        "multiple values functions similarly as for --hidden-feats."
    ),
)
@click.option(
    "--feat-drops",
    help=(
        "Dropout of input features for each GAT layer. Passing a single value or "
        "multiple values functions similarly as for --hidden-feats."
    ),
)
@click.option(
    "--attn-drops",
    help=(
        "Dropout of attention values for each GAT layer. Passing a single value or "
        "multiple values functions similarly as for --hidden-feats."
    ),
)
@click.option(
    "--alphas",
    help=(
        "Hyperparameter for LeakyReLU gate for each GAT layer. Passing a single value "
        "or multiple values functions similarly as for --hidden-feats."
    ),
)
@click.option(
    "--residuals",
    help=(
        "Whether to use residual connection for each GAT layer. Passing a single value "
        "or multiple values functions similarly as for --hidden-feats."
    ),
)
@click.option(
    "--agg-modes",
    help=(
        "Which aggregation mode [flatten, mean] to use for each GAT layer. Passing a "
        "single value or multiple values functions similarly as for --hidden-feats."
    ),
)
@click.option(
    "--biases",
    help=(
        "Whether to use bias for each GAT layer. Passing a single value "
        "or multiple values functions similarly as for --hidden-feats."
    ),
)
@click.option(
    "--allow-zero-in-degree",
    type=bool,
    help="Allow zero in degree nodes for all graph layers.",
)
# SchNet-specific parameters
@click.option("--hidden-channels", type=int, help="Hidden embedding size.")
@click.option(
    "--num-filters",
    type=int,
    help="Number of filters to use in the cfconv layers.",
)
@click.option("--num-interactions", type=int, help="Number of interaction blocks.")
@click.option(
    "--num-gaussians",
    type=int,
    help="Number of gaussians to use in the interaction blocks.",
)
@click.option(
    "--cutoff",
    type=float,
    help="Cutoff distance for interatomic interactions.",
)
@click.option(
    "--max-num-neighbors",
    type=int,
    help="Maximum number of neighbors to collect for each node.",
)
@click.option(
    "--readout",
    type=click.Choice(["add", "mean"]),
    help="Which global aggregation to use [add, mean].",
)
@click.option(
    "--dipole",
    type=bool,
    help=(
        "Whether to use the magnitude of the dipole moment to make the final "
        "prediction."
    ),
)
@click.option(
    "--mean",
    type=float,
    help=(
        "Mean of property to predict, to be added to the model prediction before "
        "returning. This value is only used if dipole is False and a value is also "
        "passed for --std."
    ),
)
@click.option(
    "--std",
    type=float,
    help=(
        "Standard deviation of property to predict, used to scale the model "
        "prediction before returning. This value is only used if dipole is False "
        "and a value is also passed for --mean."
    ),
)
# e3nn-specific parameters
@click.option(
    "--num-atom-types",
    type=int,
    help=(
        "Number of different atom types. In general, this will just be the "
        "max atomic number of all input atoms."
    ),
)
@click.option(
    "--irreps-hidden",
    help="Irreps for the hidden layers of the network.",
)
@click.option(
    "--lig", type=bool, help="Include ligand labels as a node attribute information."
)
@click.option(
    "--irreps-edge-attr",
    type=int,
    help=(
        "Which level of spherical harmonics to use for encoding edge attributes "
        "internally."
    ),
)
@click.option("--num-layers-schnet", type=int, help="Number of network layers.")
@click.option(
    "--neighbor-dist",
    type=float,
    help="Cutoff distance for including atoms as neighbors.",
)
@click.option(
    "--num-basis",
    type=int,
    help="Number of bases on which the edge length are projected.",
)
@click.option("--num-radial-layers", type=int, help="Number of radial layers.")
@click.option(
    "--num-radial-neurons",
    type=int,
    help="Number of neurons in each radial layer.",
)
@click.option("--num-neighbors", type=float, help="Typical number of neighbor nodes.")
@click.option("--num-nodes", type=float, help="Typical number of nodes in a graph.")
def test(
    output_dir: Path,
    model_type: MLModelType,
    config_file: Path | None = None,
    use_wandb: bool = False,
    sweep: bool = False,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    extra_config: list[str] | None = None,
    grouped: bool | None = None,
    strategy: ascfg.MTENNStrategy | None = None,
    pred_readout: ascfg.MTENNReadout | None = None,
    combination: ascfg.MTENNCombination | None = None,
    comb_readout: ascfg.MTENNReadout | None = None,
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
        case MLModelType.GAT:
            config_class = ascfg.GATModelConfig
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
        case MLModelType.schnet:
            config_class = ascfg.SchNetModelConfig
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
        case MLModelType.e3nn:
            config_class = ascfg.E3NNModelConfig
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
