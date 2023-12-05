from pathlib import Path

import click
from asapdiscovery.data.utils import MOONSHOT_CDD_ID_REGEX, MPRO_ID_REGEX
from asapdiscovery.ml.schema_v2.config import OptimizerType
from mtenn.config import (
    CombinationConfig,
    DatasetSplitterType,
    EarlyStoppingType,
    ReadoutConfig,
    StrategyConfig,
)


################################################################################
# IO args
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
# Optimizer args
def optim_args(func):
    for fn in [optimizer_type, lr, weight_decay, momentum, dampening, b1, b2, eps, rho]:
        func = fn(func)

    return func


def optimizer_type(func):
    return click.option(
        "-optim",
        "--optimizer-type",
        type=OptimizerType,
        help=(
            "Type of optimizer to use. "
            f"Options are [{', '.join(OptimizerType.get_values())}]."
        ),
    )(func)


# Common parameters
def lr(func):
    return click.option("--lr", type=float, help="Optimizer learning rate.")(func)


def weight_decay(func):
    return click.option(
        "--weight-decay", type=float, help="Optimizer weight decay (L2 penalty)."
    )(func)


# SGD-only parameters
def momentum(func):
    return click.option("--momentum", type=float, help="Momentum for SGD optimizer.")(
        func
    )


def dampening(func):
    return click.option(
        "--dampening", type=float, help="Dampening for momentum for SGD optimizer."
    )(func)


# Adam* parameters
def b1(func):
    return click.option(
        "--b1", type=float, help="B1 parameter for Adam and AdamW optimizers."
    )(func)


def b2(func):
    return click.option(
        "--b2", type=float, help="B2 parameter for Adam and AdamW optimizers."
    )(func)


def eps(func):
    return click.option(
        "--eps",
        type=float,
        help="Epsilon parameter for Adam, AdamW, and Adadelta optimizers.",
    )(func)


# Adadelta parameters
def rho(func):
    return click.option(
        "--rho", type=float, help="Rho parameter for Adadelta optimizer."
    )(func)


################################################################################


################################################################################
# Model setup args
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
# W&B args
def wandb_args(func):
    for fn in [use_wandb, sweep, proj, name, extra_config]:
        func = fn(func)

    return func


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
# MTENN args
def mtenn_args(func):
    for fn in [
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
    ]:
        func = fn(func)

    return func


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
# GAT args
def gat_args(func):
    for fn in [
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
    ]:
        func = fn(func)

    return func


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
# SchNet args
def schnet_args(func):
    for fn in [
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
    ]:
        func = fn(func)

    return func


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
# E3NN args
def e3nn_args(func):
    for fn in [
        num_atom_types,
        irreps_hidden,
        lig,
        irreps_edge_attr,
        num_layers_e3nn,
        neighbor_dist,
        num_basis,
        num_radial_layers,
        num_radial_neurons,
        num_neighbors,
        num_nodes,
    ]:
        func = fn(func)

    return func


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


def num_layers_e3nn(func):
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


################################################################################
# Early stopping args
def es_args(func):
    for fn in [es_type, es_patience, es_n_check, es_divergence, es_config_cache]:
        func = fn(func)

    return func


def es_type(func):
    return click.option(
        "--es-type",
        type=EarlyStoppingType,
        help=(
            "Which early stopping strategy to use. "
            f"Options are [{', '.join(OptimizerType.get_values())}]."
        ),
    )(func)


def es_patience(func):
    return click.option(
        "--es-patience",
        type=int,
        help=(
            "Number of training epochs to allow with no improvement in val loss. "
            "Used if --es_type is best."
        ),
    )(func)


def es_n_check(func):
    return click.option(
        "--es-n-check",
        type=int,
        help=(
            "Number of past epoch losses to keep track of when determining "
            "convergence. Used if --es_type is converged."
        ),
    )(func)


def es_divergence(func):
    return click.option(
        "--es-divergence",
        type=float,
        help=(
            "Max allowable difference from the mean of the losses as a fraction of the "
            "average loss. Used if --es_type is converged."
        ),
    )(func)


def es_config_cache(func):
    return click.option(
        "--es-config-cache",
        type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
        help=(
            "EarlyStoppingConfig JSON cache file. Other early stopping-related args "
            "that are passed will supersede anything stored in this file."
        ),
    )(func)


################################################################################


################################################################################
# Dataset args
def exp_file(func):
    return click.option(
        "-exp",
        "--exp-file",
        type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
        help="JSON file giving a list of ExperimentalDataCompound objects.",
    )(func)


def str_files(func):
    return click.option(
        "-str",
        "--structures",
        type=str,
        help=(
            "PDB structure files. Can be in one of two forms: either a glob that will "
            "be expanded and all matching files will be taken, or a directory, in "
            "which case all top-level PDB files will be taken."
        ),
    )(func)


def str_fn_xtal_regex(func):
    return click.option(
        "--xtal-regex",
        default=MPRO_ID_REGEX,
        help="Regex for extracting crystal structure name from filename.",
    )(func)


def str_fn_cpd_regex(func):
    return click.option(
        "--cpd-regex",
        default=MOONSHOT_CDD_ID_REGEX,
        help="Regex for extracting compound id from filename.",
    )(func)


def ds_cache(func):
    return click.option(
        "--ds-cache",
        type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
        help="Dataset cache file.",
    )(func)


def ds_config_cache(func):
    return click.option(
        "--ds-config-cache",
        type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
        help=(
            "DatasetConfig JSON cache file. If this is given, no other dataset-related "
            "args will be parsed."
        ),
    )(func)


################################################################################


################################################################################
# Dataset splitter args
def ds_split_args(func):
    for fn in [
        ds_split_type,
        train_frac,
        val_frac,
        test_frac,
        enforce_1,
        rand_seed,
        ds_split_config_cache,
    ]:
        func = fn(func)

    return func


def ds_split_type(func):
    return click.option(
        "--ds-split-type",
        type=DatasetSplitterType,
        help=(
            "Method to use for splitting. "
            f"Options are [{', '.join(DatasetSplitterType.get_values())}]."
        ),
    )(func)


def train_frac(func):
    return click.option(
        "--train-frac",
        type=float,
        help="Fraction of dataset to put in the train split.",
    )(func)


def val_frac(func):
    return click.option(
        "--val-frac",
        type=float,
        help="Fraction of dataset to put in the val split.",
    )(func)


def test_frac(func):
    return click.option(
        "--test-frac",
        type=float,
        help="Fraction of dataset to put in the test split.",
    )(func)


def enforce_1(func):
    return click.option(
        "--enforce-one",
        type=bool,
        help="Make sure that all split fractions add up to 1.",
    )(func)


def rand_seed(func):
    return click.option(
        "--rand-seed", type=int, help="Random seed to use if randomly splitting data."
    )(func)


def ds_split_config_cache(func):
    return click.option(
        "--ds-split-config-cache",
        type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
        help=(
            "DatasetSplitterConfig JSON cache file. Other dataset splitter-related "
            "args that are passed will supersede anything stored in this file."
        ),
    )(func)


################################################################################
