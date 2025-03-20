from pathlib import Path

import click
import torch
from asapdiscovery.data.util.utils import MOONSHOT_CDD_ID_REGEX, MPRO_ID_REGEX
from asapdiscovery.ml.config import (
    DatasetSplitterType,
    EarlyStoppingType,
    OptimizerType,
)
from mtenn.config import CombinationConfig, ReadoutConfig, StrategyConfig


################################################################################
# IO args
def output_dir(func):
    return click.option(
        "-o",
        "--output-dir",
        type=click.Path(
            exists=False, file_okay=False, dir_okay=True, writable=True, path_type=Path
        ),
        help=(
            "Top-level output directory. A subdirectory with the current W&B "
            "run ID will be made/searched if W&B is being used."
        ),
    )(func)


def save_weights(func):
    return click.option(
        "--save-weights",
        type=click.Choice(["all", "recent", "final"], case_sensitive=False),
        help=(
            "How often to save weights during training."
            'Options are to keep every epoch ("all"), only keep the most recent '
            'epoch ("recent"), or only keep the final epoch ("final").'
        ),
    )(func)


def model_tag(func):
    return click.option(
        "--model-tag",
        type=str,
        help="Tag to name model weights files when saving.",
    )(func)


def trainer_config_cache(func):
    return click.option(
        "--trainer-config-cache",
        type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
        help=(
            "Trainer Config JSON cache file. Any other CLI args that are passed will "
            "supersede anything in this file."
        ),
    )(func)


def sweep_config_cache(func):
    return click.option(
        "--sweep-config-cache",
        type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
        help=(
            "Sweeper Config JSON cache file. If this file exists, no other CLI args "
            "will be parsed."
        ),
    )(func)


################################################################################

# S3 args


def s3_path(func):
    return click.option(
        "--s3-path",
        type=str,
        help="S3 path to store the results.",
    )(func)


def upload_to_s3(func):
    return click.option(
        "--upload-to-s3",
        is_flag=True,
        help="Whether to upload the results to S3.",
    )(func)


def s3_args(func):
    for fn in [s3_path, upload_to_s3]:
        func = fn(func)

    return func


################################################################################
# Overwrite flags
def overwrite_args(func):
    for fn in [
        trainer_config_cache_overwrite,
        optimizer_config_cache_overwrite,
        model_config_cache_overwrite,
        es_config_cache_overwrite,
        ds_config_cache_overwrite,
        ds_cache_overwrite,
        ds_split_config_cache_overwrite,
    ]:
        func = fn(func)

    return func


def trainer_config_cache_overwrite(func):
    return click.option(
        "--overwrite-trainer-config-cache",
        is_flag=True,
        help="Overwrite any existing Trainer JSON cache file.",
    )(func)


def sweep_config_cache_overwrite(func):
    return click.option(
        "--overwrite-sweep-config-cache",
        is_flag=True,
        help="Overwrite any existing Sweeper JSON cache file.",
    )(func)


def optimizer_config_cache_overwrite(func):
    return click.option(
        "--overwrite-optimizer-config-cache",
        is_flag=True,
        help="Overwrite any existing OptimzerConfig JSON cache file.",
    )(func)


def model_config_cache_overwrite(func):
    return click.option(
        "--overwrite-model-config-cache",
        is_flag=True,
        help="Overwrite any existing ModelConfig JSON cache file.",
    )(func)


def es_config_cache_overwrite(func):
    return click.option(
        "--overwrite-es-config-cache",
        is_flag=True,
        help="Overwrite any existing EarlyStoppingConfig JSON cache file.",
    )(func)


def ds_config_cache_overwrite(func):
    return click.option(
        "--overwrite-ds-config-cache",
        is_flag=True,
        help="Overwrite any existing DatasetConfig JSON cache file.",
    )(func)


def ds_cache_overwrite(func):
    return click.option(
        "--overwrite-ds-cache",
        is_flag=True,
        help="Overwrite any existing Dataset pkl cache file.",
    )(func)


def ds_split_config_cache_overwrite(func):
    return click.option(
        "--overwrite-ds-split-config-cache",
        is_flag=True,
        help="Overwrite any existing DatasetSplitterConfig JSON cache file.",
    )(func)


################################################################################


################################################################################
# Optimizer args
def optim_args(func):
    for fn in [
        optimizer_type,
        lr,
        weight_decay,
        momentum,
        dampening,
        b1,
        b2,
        eps,
        rho,
        optimizer_config_cache,
    ]:
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


def optimizer_config_cache(func):
    return click.option(
        "--optimizer-config-cache",
        type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
        help=(
            "Optimizer Config JSON cache file. Other optimizer-related args "
            "that are passed will supersede anything stored in this file."
        ),
    )(func)


################################################################################


################################################################################
# Model setup args
def model_config_cache(func):
    return click.option(
        "--model-config-cache",
        type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
        help=(
            "Model Config JSON cache file. Other model-related args "
            "that are passed will supersede anything stored in this file."
        ),
    )(func)


def model_rand_seed(func):
    return click.option(
        "--model-rand-seed", type=int, help="Random seed for initializing the model."
    )(func)


def weights_path(func):
    return click.option(
        "--weights-path",
        type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
        help=(
            "Path to an existing weights file. Use this for loading pretrained "
            "weights from a previous run as the starting weights."
        ),
    )(func)


################################################################################


################################################################################
# W&B args
def wandb_args(func):
    for fn in [use_wandb, proj, name, extra_config]:
        func = fn(func)

    return func


def use_wandb(func):
    return click.option(
        "--use-wandb", type=bool, help="Use W&B to log model training."
    )(func)


def proj(func):
    return click.option("-proj", "--wandb-project", help="W&B project name.")(func)


def name(func):
    return click.option("-name", "--wandb-name", help="W&B run name.")(func)


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
        strategy_layer_norm,
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


def strategy_layer_norm(func):
    return click.option(
        "--strategy-layer-norm",
        type=bool,
        help="Apply a LayerNorm operation in the Strategy.",
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
# order in cli_args need to match order in cli.py
def visnet_args(func):
    for fn in [
        num_heads,
        lmax,
        vecnorm_type,
        trainable_vecnorm,
        num_layers_visnet,
        hidden_channels,
        num_rbf,
        trainable_rbf,
        max_z,
        cutoff,
        max_num_neighbors,
        vertex,
        reduce_op,
        mean,
        std,
        derivative,
    ]:
        func = fn(func)

    return func


def lmax(func):
    return click.option(
        "--lmax",
        type=int,
        help=("The maximum degree of the spherical harmonics"),
    )(func)


def vecnorm_type(func):
    return click.option(
        "--vecnorm-type",
        type=str,
        help=("Type of vector normalization to use. ['max_min', None]"),
    )(func)


def trainable_vecnorm(func):
    return click.option(
        "--trainable-vecnorm",
        type=bool,
        help="Whether to make the vector normalization trainable.",
    )(func)


def num_layers_visnet(func):
    return click.option(
        "--num-layers",
        type=int,
        help="Number of network layers.",
    )(func)


def num_rbf(func):
    return click.option(
        "--num-rbf",
        type=int,
        help="Number of radial basis functions.",
    )(func)


def trainable_rbf(func):
    return click.option(
        "--trainable-rbf",
        type=bool,
        help="Whether to make the radial basis functions trainable.",
    )(func)


def max_z(func):
    return click.option(
        "--max-z",
        type=int,
        help="Maximum atomic number.",
    )(func)


def vertex(func):
    return click.option(
        "--vertex",
        type=bool,
        help="Whether to use the vertex geometric features.",
    )(func)


def reduce_op(func):
    return click.option(
        "--reduce-op",
        type=str,
        help="Reduce operation. ['sum', 'mean']",
    )(func)


def derivative(func):
    return click.option(
        "--derivative",
        type=bool,
        help="Whether to use the derivative. NOT USED.",
    )(func)


################################################################################


################################################################################
# Early stopping args
def es_args(func):
    for fn in [
        es_type,
        es_patience,
        es_n_check,
        es_divergence,
        es_burnin,
        es_config_cache,
    ]:
        func = fn(func)

    return func


def es_type(func):
    return click.option(
        "--es-type",
        type=EarlyStoppingType,
        help=(
            "Which early stopping strategy to use. "
            f"Options are [{', '.join(EarlyStoppingType.get_values())}]."
        ),
    )(func)


def es_patience(func):
    return click.option(
        "--es-patience",
        type=int,
        help=(
            "Number of training epochs to allow with no improvement in val loss. "
            "Used if --es_type is best or patient_converged."
        ),
    )(func)


def es_n_check(func):
    return click.option(
        "--es-n-check",
        type=int,
        help=(
            "Number of past epoch losses to keep track of when determining "
            "convergence. Used if --es_type is converged or patient_converged."
        ),
    )(func)


def es_divergence(func):
    return click.option(
        "--es-divergence",
        type=float,
        help=(
            "Max allowable difference from the mean of the losses as a fraction of the "
            "average loss. Used if --es_type is converged or patient_converged."
        ),
    )(func)


def es_burnin(func):
    return click.option(
        "--es-burnin",
        type=int,
        help=(
            "Minimum number of epochs to train for regardless of early "
            "stopping criteria."
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
def graph_ds_args(func):
    for fn in [exp_file, ds_cache, ds_config_cache]:
        func = fn(func)

    return func


def struct_ds_args(func):
    for fn in [str_files, str_fn_xtal_regex, str_fn_cpd_regex]:
        func = fn(func)

    return func


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
        enforce_one,
        ds_rand_seed,
        ds_split_dict,
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


def enforce_one(func):
    return click.option(
        "--enforce-one",
        type=bool,
        help="Make sure that all split fractions add up to 1.",
    )(func)


def ds_rand_seed(func):
    return click.option(
        "--ds-rand-seed",
        type=int,
        help="Random seed to use if randomly splitting data.",
    )(func)


def ds_split_dict(func):
    return click.option(
        "--ds-split-dict",
        type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
        help=(
            "JSON file containing the split dict to use in the case of manual "
            'splitting. The dict should map the keys ["train", "val", "test"] '
            "to lists containing the compounds that belong in each split."
        ),
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


################################################################################
# Loss function args
def loss_args(func):
    for fn in [loss, loss_weights, eval_loss_weights]:
        func = fn(func)

    return func


def loss(func):
    return click.option(
        "--loss",
        type=str,
        multiple=True,
        help=(
            "Specifications for loss function(s) to use. Multiple can be passed, and "
            "they will be weighted as specified with --loss-weights. Each individual "
            "loss function should be specified as a comma separated list of "
            "<key>:<value> pairs, which will be passed directly to the "
            "LossFunctionConfig class. For example, to add a loss term that penalizes "
            "predictions for being outside a normal pIC50 range, you could pass "
            "--loss loss_type:range,range_lower_lim:0,range_upper_lim:10."
        ),
    )(func)


def loss_weights(func):
    return click.option(
        "--loss-weights",
        type=float,
        multiple=True,
        help=(
            "Weights for each loss function. If no weights values are passed, each "
            "loss term will be weighted equally. These args are assumed to be in the "
            "same order as the --loss args that they correspond to."
        ),
    )(func)


def eval_loss_weights(func):
    return click.option(
        "--eval-loss-weights",
        type=float,
        multiple=True,
        help=(
            "Weights for each loss function for val and test sets. If no values are "
            "passed, will reuse the values from --loss-weights. These args are assumed "
            "to be in the same order as the --loss args that they correspond to."
        ),
    )(func)


################################################################################


################################################################################
# Training args
def trainer_args(func):
    for fn in [
        start_epoch,
        n_epochs,
        batch_size,
        target_prop,
        cont,
        loss_dict,
        device,
        data_aug,
        trainer_weight_decay,
        batch_norm,
    ]:
        func = fn(func)
    return func


def start_epoch(func):
    return click.option(
        "--start-epoch",
        type=int,
        help="Which epoch to start training at (used for continuing training runs).",
    )(func)


def n_epochs(func):
    return click.option(
        "--n-epochs",
        type=int,
        help=(
            "Which epoch to stop training at. For non-continuation runs, this "
            "will be the total number of epochs to train for."
        ),
    )(func)


def batch_size(func):
    return click.option(
        "--batch-size",
        type=int,
        help="Number of samples to predict on before performing backprop.",
    )(func)


def target_prop(func):
    return click.option(
        "--target-prop", type=str, help="Target property to train against."
    )(func)


def cont(func):
    return click.option(
        "--cont",
        type=bool,
        help="This is a continuation of a previous training run.",
    )(func)


def loss_dict(func):
    return click.option(
        "--loss-dict",
        type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
        help=(
            "JSON file storing the dict of losses. Use in continuation runs. If not "
            "given during a continuation run, loss_dict.json will be loaded from the "
            "provided output-dir."
        ),
    )(func)


def device(func):
    return click.option("--device", type=torch.device, help="Device to train on.")(func)


def data_aug(func):
    return click.option(
        "--data-aug",
        type=str,
        multiple=True,
        help=(
            "Specifications for data augmentations to do. Multiple can be passed, and "
            "they will be applied in the order they are specified on the command line. "
            "Each individual aug config should be specified as a comma separated list "
            "of <key>:<value> pairs, which will be passed directly to the "
            "DataAugConfig class. For example, to add positional jittering that draws "
            "noise from a fixed Gaussian with a std of 0.05, you would pass "
            "--data-aug aug_type:jitter_fixed,jitter_fixed_std:0.05."
        ),
    )(func)


def trainer_weight_decay(func):
    return click.option(
        "--trainer-weight-decay",
        type=float,
        help=(
            "Weight decay weighting for training. This will add a term of "
            "weight_decay / 2 * the square of the L2-norm of the model weights, "
            "excluding any bias terms."
        ),
    )(func)


def batch_norm(func):
    return click.option(
        "--batch-norm", type=bool, help="Normalize batch gradient by batch size."
    )(func)


################################################################################


################################################################################
# Sweep args
def sweep_args(func):
    for fn in [sweep_config, force_new_sweep, sweep_start_only]:
        func = fn(func)
    return func


def sweep_config(func):
    return click.option(
        "--sweep-config",
        type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
        help="YAML file giving the config for a sweep.",
    )(func)


def force_new_sweep(func):
    return click.option(
        "--force-new-sweep",
        type=bool,
        help="Start a new sweep even if an existing sweep_id is present.",
    )(func)


def sweep_start_only(func):
    return click.option(
        "--start-only",
        type=bool,
        is_flag=True,
        default=False,
        help="Only start the sweep, don't run any training.",
    )(func)


################################################################################


################################################################################
# Helper functions
def kvp_list_to_dict(kvp_list_str):
    """
    Convert a string that consists of a comma-separated list of <key>:<value> pairs into
    a dict.

    Parameters
    ----------
    kvp_list_str : str
        String from CLI containing key:value pairs

    Returns
    -------
    dict
        Python dict built from the input string
    """

    return {kvp.split(":")[0]: kvp.split(":")[1] for kvp in kvp_list_str.split(",")}


################################################################################
