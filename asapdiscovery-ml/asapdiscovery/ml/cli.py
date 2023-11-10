import click

import asapdiscovery.ml.schema_v2.config as ascfg


@click.group()
def ml():
    pass


@ml.command()
@click.option("-model", "--model-type", required=True, type=ascfg.ModelType)
def test(model_type: ascfg.ModelType):
    # Build the model
    match model_type:
        case ascfg.ModelType.gat:
            config = ascfg.GATModelConfig()
        case ascfg.ModelType.schnet:
            config = ascfg.SchNetModelConfig()
        case ascfg.ModelType.e3nn:
            config = ascfg.E3NNModelConfig()
        case unknown:
            raise ValueError(f"Unknown model type: {unknown}")
    model = config.build()
