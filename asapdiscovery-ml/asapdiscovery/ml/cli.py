import asapdiscovery.ml.schema_v2.config as ascfg
import click


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
    model = config.build()
    print(model, flush=True)
