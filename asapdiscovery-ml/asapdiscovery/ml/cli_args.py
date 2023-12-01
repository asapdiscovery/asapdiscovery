import click
from pathlib import Path

from mtenn.config import (
    CombinationConfig,
    E3NNModelConfig,
    GATModelConfig,
    ModelType,
    ReadoutConfig,
    SchNetModelConfig,
    StrategyConfig,
)


################################################################################
## IO args
def output_dir(func):
    return click.option(
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
    )(func)


################################################################################


################################################################################
## Model setup args
def config_file(func):
    return click.option(
        "--config-file",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help=(
            "JSON file giving model config. Any passed CLI args will overwrite the "
            "options in this file."
        ),
    )(func)


################################################################################


################################################################################
## W&B args
def wandb_args(func):
    for fn in [use_wandb, sweep, proj, name, extra_config]:
        func = fn(func)


def use_wandb(func):
    return click.option(
        "--use-wandb", is_flag=True, help="Use W&B to log model training."
    )(func)


def sweep(func):
    return click.option(
        "--sweep", is_flag=True, help="This run is part of a W&B sweep."
    )(func)


def proj(func):
    return click.option("-proj", "--wandb-project", help="W&B project name.")(func)


def name(func):
    return click.option("-name", "--wandb-name", help="W&B project name.")(func)


def extra_config(func):
    return click.option(
        "-e",
        "--extra-config",
        multiple=True,
        help=(
            "Any extra config options to log to W&B, provided as comma-separated pairs. "
            "Can be provided as many times as desired "
            "(eg -e key1,val1 -e key2,val2 -e key3,val3)."
        ),
    )(func)


################################################################################


################################################################################
## MTENN args
def grouped(func):
    return click.option(
        "--grouped",
        type=bool,
        help="Model is a grouped (multi-pose) model.",
    )(func)


def strategy(func):
    return click.option(
        "--strategy",
        type=StrategyConfig,
        help=(
            "Which Strategy to use for combining complex, protein, and ligand "
            "representations in the MTENN Model."
        ),
    )(func)


def pred_readout(func):
    return click.option(
        "--pred-readout",
        type=ReadoutConfig,
        help=(
            "Which Readout to use for the model predictions. This corresponds "
            "to the individual pose predictions in the case of a GroupedModel."
        ),
    )(func)


def combination(func):
    return click.option(
        "--combination",
        type=CombinationConfig,
        help="Which Combination to use for combining predictions in a GroupedModel.",
    )(func)


def comb_readout(func):
    return click.option(
        "--comb-readout",
        type=ReadoutConfig,
        help=(
            "Which Readout to use for the combined model predictions. This is only "
            "relevant in the case of a GroupedModel."
        ),
    )(func)


def max_comb_neg(func):
    return click.option(
        "--max-comb-neg",
        type=bool,
        help=(
            "Whether to take the min instead of max when combining pose predictions "
            "with MaxCombination."
        ),
    )(func)


def max_comb_scale(func):
    return click.option(
        "--max-comb-scale",
        type=float,
        help=(
            "Scaling factor for values when taking the max/min when combining pose "
            "predictions with MaxCombination. A value of 1 will approximate the "
            "Boltzmann mean, while a larger value will more accurately approximate the "
            "max/min operation."
        ),
    )(func)


def pred_substrate(func):
    return click.option(
        "--pred-substrate",
        type=float,
        help=(
            "Substrate concentration to use when using the Cheng-Prusoff equation to "
            "convert deltaG -> IC50 in PIC50Readout for pred_readout. Assumed to be in "
            "the same units as pred_km."
        ),
    )(func)


def pred_km(func):
    return click.option(
        "--pred-km",
        type=float,
        help=(
            "Km value to use when using the Cheng-Prusoff equation to convert "
            "deltaG -> IC50 in PIC50Readout for pred_readout. Assumed to be in "
            "the same units as pred_substrate."
        ),
    )(func)


def comb_substrate(func):
    return click.option(
        "--comb-substrate",
        type=float,
        help=(
            "Substrate concentration to use when using the Cheng-Prusoff equation to "
            "convert deltaG -> IC50 in PIC50Readout for comb_readout. Assumed to be in "
            "the same units as comb_km."
        ),
    )(func)


def comb_km(func):
    return click.option(
        "--comb-km",
        type=float,
        help=(
            "Km value to use when using the Cheng-Prusoff equation to convert "
            "deltaG -> IC50 in PIC50Readout for comb_readout. Assumed to be in "
            "the same units as comb_substrate."
        ),
    )(func)


################################################################################


################################################################################
## GAT args
def in_feats(func):
    return click.option("--in-feats", type=int, help="Input node feature size.")(func)


def num_layers_gat(func):
    return click.option(
        "--num-layers",
        type=int,
        help=(
            "Number of GAT layers. Ignored if multiple values are passed for any "
            "other GAT argument. To define a model with only one layer, this must be "
            "explicitly set to 1."
        ),
    )(func)


def hidden_feats(func):
    return click.option(
        "--hidden-feats",
        help=(
            "Output size of each GAT layer. This can either be a single value, which will "
            "be broadcasted to each layer, or a comma-separated list with each value "
            "corresponding to one layer in the model."
        ),
    )(func)


def num_heads(func):
    return click.option(
        "--num-heads",
        help=(
            "Number of attention heads for each GAT layer. Passing a single value or "
            "multiple values functions similarly as for --hidden-feats."
        ),
    )(func)


def feat_drops(func):
    return click.option(
        "--feat-drops",
        help=(
            "Dropout of input features for each GAT layer. Passing a single value or "
            "multiple values functions similarly as for --hidden-feats."
        ),
    )(func)


def attn_drops(func):
    return click.option(
        "--attn-drops",
        help=(
            "Dropout of attention values for each GAT layer. Passing a single value or "
            "multiple values functions similarly as for --hidden-feats."
        ),
    )(func)


def alphas(func):
    return click.option(
        "--alphas",
        help=(
            "Hyperparameter for LeakyReLU gate for each GAT layer. Passing a single value "
            "or multiple values functions similarly as for --hidden-feats."
        ),
    )(func)


def residuals(func):
    return click.option(
        "--residuals",
        help=(
            "Whether to use residual connection for each GAT layer. Passing a single value "
            "or multiple values functions similarly as for --hidden-feats."
        ),
    )(func)


def agg_modes(func):
    return click.option(
        "--agg-modes",
        help=(
            "Which aggregation mode [flatten, mean] to use for each GAT layer. Passing a "
            "single value or multiple values functions similarly as for --hidden-feats."
        ),
    )(func)


def biases(func):
    return click.option(
        "--biases",
        help=(
            "Whether to use bias for each GAT layer. Passing a single value "
            "or multiple values functions similarly as for --hidden-feats."
        ),
    )(func)


def allow_zero_in_degree(func):
    return click.option(
        "--allow-zero-in-degree",
        type=bool,
        help="Allow zero in degree nodes for all graph layers.",
    )(func)


################################################################################


################################################################################
## SchNet args
def hidden_channels(func):
    return click.option("--hidden-channels", type=int, help="Hidden embedding size.")(
        func
    )


def num_filters(func):
    return click.option(
        "--num-filters",
        type=int,
        help="Number of filters to use in the cfconv layers.",
    )(func)


def num_interactions(func):
    return click.option(
        "--num-interactions", type=int, help="Number of interaction blocks."
    )(func)


def num_gaussians(func):
    return click.option(
        "--num-gaussians",
        type=int,
        help="Number of gaussians to use in the interaction blocks.",
    )(func)


def cutoff(func):
    return click.option(
        "--cutoff",
        type=float,
        help="Cutoff distance for interatomic interactions.",
    )(func)


def max_num_neighbors(func):
    return click.option(
        "--max-num-neighbors",
        type=int,
        help="Maximum number of neighbors to collect for each node.",
    )(func)


def readout(func):
    return click.option(
        "--readout",
        type=click.Choice(["add", "mean"]),
        help="Which global aggregation to use [add, mean].",
    )(func)


def dipole(func):
    return click.option(
        "--dipole",
        type=bool,
        help=(
            "Whether to use the magnitude of the dipole moment to make the final "
            "prediction."
        ),
    )(func)


def mean(func):
    return click.option(
        "--mean",
        type=float,
        help=(
            "Mean of property to predict, to be added to the model prediction before "
            "returning. This value is only used if dipole is False and a value is also "
            "passed for --std."
        ),
    )(func)


def std(func):
    return click.option(
        "--std",
        type=float,
        help=(
            "Standard deviation of property to predict, used to scale the model "
            "prediction before returning. This value is only used if dipole is False "
            "and a value is also passed for --mean."
        ),
    )(func)


################################################################################


################################################################################
## E3NN args
def num_atom_types(func):
    return click.option(
        "--num-atom-types",
        type=int,
        help=(
            "Number of different atom types. In general, this will just be the "
            "max atomic number of all input atoms."
        ),
    )(func)


def irreps_hidden(func):
    return click.option(
        "--irreps-hidden",
        help="Irreps for the hidden layers of the network.",
    )(func)


def lig(func):
    return click.option(
        "--lig",
        type=bool,
        help="Include ligand labels as a node attribute information.",
    )(func)


def irreps_edge_attr(func):
    return click.option(
        "--irreps-edge-attr",
        type=int,
        help=(
            "Which level of spherical harmonics to use for encoding edge attributes "
            "internally."
        ),
    )(func)


def num_layers_schnet(func):
    return click.option("--num-layers", type=int, help="Number of network layers.")(
        func
    )


def neighbor_dist(func):
    return click.option(
        "--neighbor-dist",
        type=float,
        help="Cutoff distance for including atoms as neighbors.",
    )(func)


def num_basis(func):
    return click.option(
        "--num-basis",
        type=int,
        help="Number of bases on which the edge length are projected.",
    )(func)


def num_radial_layers(func):
    return click.option(
        "--num-radial-layers", type=int, help="Number of radial layers."
    )(func)


def num_radial_neurons(func):
    return click.option(
        "--num-radial-neurons",
        type=int,
        help="Number of neurons in each radial layer.",
    )(func)


def num_neighbors(func):
    return click.option(
        "--num-neighbors", type=float, help="Typical number of neighbor nodes."
    )(func)


def num_nodes(func):
    return click.option(
        "--num-nodes", type=float, help="Typical number of nodes in a graph."
    )(func)


################################################################################
