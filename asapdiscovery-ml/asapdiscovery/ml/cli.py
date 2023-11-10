import asapdiscovery.ml.schema_v2.config as ascfg
import click
from dgllife.utils import CanonicalAtomFeaturizer


@click.group()
def ml():
    pass


@ml.command()
@click.option(
    "-model",
    "--model-type",
    required=True,
    type=ascfg.ModelType,
    help="Which model type to use.",
)
# Shared MTENN-related parameters
@click.option(
    "--grouped",
    is_flag=True,
    default=False,
    help="Model is a grouped (multi-pose) model.",
)
@click.option(
    "--strategy",
    type=ascfg.MTENNStrategy,
    default=ascfg.MTENNStrategy.delta,
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
    default=True,
    help=(
        "Whether to take the min instead of max when combining pose predictions "
        "with MaxCombination."
    ),
)
@click.option(
    "--max-comb-scale",
    type=float,
    default=1000,
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
@click.option(
    "--in-feats",
    type=int,
    default=CanonicalAtomFeaturizer().feat_size(),
    help=("Input node feature size. Defaults to size of the CanonicalAtomFeaturizer."),
)
@click.option(
    "--num-layers",
    type=int,
    default=2,
    help=(
        "Number of GAT layers. Ignored if multiple values are passed for any "
        "other GAT argument. To define a model with only one layer, this must be "
        "explicitly set to 1."
    ),
)
@click.option(
    "--hidden-feats",
    default="32",
    help=(
        "Output size of each GAT layer. This can either be a single value, which will "
        "be broadcasted to each layer, or a comma-separated list with each value "
        "corresponding to one layer in the model."
    ),
)
@click.option(
    "--num-heads",
    default="4",
    help=(
        "Number of attention heads for each GAT layer. Passing a single value or "
        "multiple values functions similarly as for --hidden-feats."
    ),
)
@click.option(
    "--feat-drops",
    default="0",
    help=(
        "Dropout of input features for each GAT layer. Passing a single value or "
        "multiple values functions similarly as for --hidden-feats."
    ),
)
@click.option(
    "--attn-drops",
    default="0",
    help=(
        "Dropout of attention values for each GAT layer. Passing a single value or "
        "multiple values functions similarly as for --hidden-feats."
    ),
)
@click.option(
    "--alphas",
    default="0.2",
    help=(
        "Hyperparameter for LeakyReLU gate for each GAT layer. Passing a single value "
        "or multiple values functions similarly as for --hidden-feats."
    ),
)
@click.option(
    "--residuals",
    default="True",
    help=(
        "Whether to use residual connection for each GAT layer. Passing a single value "
        "or multiple values functions similarly as for --hidden-feats."
    ),
)
@click.option(
    "--agg-modes",
    default="flatten",
    help=(
        "Which aggregation mode [flatten, mean] to use for each GAT layer. Passing a "
        "single value or multiple values functions similarly as for --hidden-feats."
    ),
)
@click.option(
    "--biases",
    default="True",
    help=(
        "Whether to use bias for each GAT layer. Passing a single value "
        "or multiple values functions similarly as for --hidden-feats."
    ),
)
@click.option(
    "--allow-zero-in-degree",
    is_flag=True,
    help="Allow zero in degree nodes for all graph layers.",
)
def test(
    model_type: ascfg.ModelType,
    grouped: bool = False,
    strategy: ascfg.MTENNStrategy = ascfg.MTENNStrategy.delta,
    pred_readout: ascfg.MTENNReadout | None = None,
    combination: ascfg.MTENNCombination | None = None,
    comb_readout: ascfg.MTENNReadout | None = None,
    max_comb_neg: bool = True,
    max_comb_scale: float = 1000,
    pred_substrate: float | None = None,
    pred_km: float | None = None,
    comb_substrate: float | None = None,
    comb_km: float | None = None,
    in_feats: int = CanonicalAtomFeaturizer().feat_size(),
    num_layers: int = 2,
    hidden_feats: str = "32",
    num_heads: str = "4",
    feat_drops: str = "0",
    attn_drops: str = "0",
    alphas: str = "0.2",
    residuals: str = "True",
    agg_modes: str = "flatten",
    biases: str = "True",
    allow_zero_in_degree: bool = False,
):
    # Build the model
    match model_type:
        case ascfg.ModelType.gat:
            config = ascfg.GATModelConfig(
                grouped=grouped,
                strategy=strategy,
                pred_readout=pred_readout,
                combination=combination,
                comb_readout=comb_readout,
                max_comb_neg=max_comb_neg,
                max_comb_scale=max_comb_scale,
                pred_substrate=pred_substrate,
                pred_km=pred_km,
                comb_substrate=comb_substrate,
                comb_km=comb_km,
                in_feats=in_feats,
                num_layers=num_layers,
                hidden_feats=hidden_feats,
                num_heads=num_heads,
                feat_drops=feat_drops,
                attn_drops=attn_drops,
                alphas=alphas,
                residuals=residuals,
                agg_modes=agg_modes,
                biases=biases,
                allow_zero_in_degree=allow_zero_in_degree,
            )
        case ascfg.ModelType.schnet:
            config = ascfg.SchNetModelConfig(
                grouped=grouped,
                strategy=strategy,
                pred_readout=pred_readout,
                combination=combination,
                comb_readout=comb_readout,
                max_comb_neg=max_comb_neg,
                max_comb_scale=max_comb_scale,
                pred_substrate=pred_substrate,
                pred_km=pred_km,
                comb_substrate=comb_substrate,
                comb_km=comb_km,
            )
        case ascfg.ModelType.e3nn:
            config = ascfg.E3NNModelConfig(
                grouped=grouped,
                strategy=strategy,
                pred_readout=pred_readout,
                combination=combination,
                comb_readout=comb_readout,
                max_comb_neg=max_comb_neg,
                max_comb_scale=max_comb_scale,
                pred_substrate=pred_substrate,
                pred_km=pred_km,
                comb_substrate=comb_substrate,
                comb_km=comb_km,
            )
        case unknown:
            raise ValueError(f"Unknown model type: {unknown}")
    print(config, flush=True)
    model = config.build()
    print(model, flush=True)
