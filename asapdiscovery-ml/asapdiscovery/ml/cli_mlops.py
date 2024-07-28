import click
from asapdiscovery.alchemy.cli.utils import has_warhead
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.util.logging import FileLogger
from asapdiscovery.alchemy.predict import download_cdd_data
from asapdiscovery.data.util.utils import (
    cdd_to_schema,
    filter_molecules_dataframe,
    parse_fluorescence_data_cdd,
)
from shutil import rmtree

from asapdiscovery.ml.config import (
    DatasetConfig,
    DatasetSplitterConfig,
    EarlyStoppingConfig,
    LossFunctionConfig,
    OptimizerConfig,
)
from pathlib import Path
from asapdiscovery.ml.trainer import Trainer
from mtenn.config import GATModelConfig
from asapdiscovery.ml.cli_args import (
    output_dir,
)
from asapdiscovery.data.services.services_config import S3Settings
from asapdiscovery.cli.cli_args import loglevel

from openff.toolkit import Molecule
from openff.toolkit.utils.exceptions import (
    RadicalsNotSupportedError,
)
import logging
import datetime
import pandas as pd
from pathlib import Path
import yaml


# logging 
logger = logging.getLogger(__name__)

"""
TODO:
- hook up to WandB and add hyperparameter search per model in ensemble ('sweep')
"""

PROTOCOLS = [
    "MERS-CoV-MPro_fluorescence-dose-response_weizmann",
    "SARS-CoV-2-MPro_fluorescence-dose-response_weizmann",
    "EVA-71-3C_fluorescence-dose-response_weizmann",
    "EVD-68-3C_fluorescence-dose-response_weizmann",
    "DENV-1-NS2B-NS3_fluorescence-dose-response_weizmann",
    "DENV-2-NS2B-NS3_fluorescence-dose-response_weizmann",
    "DENV-3-NS2B-NS3_fluorescence-dose-response_weizmann",
    "DENV-4-NS2B-NS3_fluorescence-dose-response_weizmann",
    "WNV-NS2B-NS3_fluorescence-dose-response_weizmann",
    "ZIKV-NS2B-NS3_fluorescence-dose-response_weizmann",
]



def _train_single_model(exp_data_json, output_dir):

    logging.info(f"Training GAT model for {exp_data_json}")
    optimizer_config = OptimizerConfig()
    gat_model_config = GATModelConfig()
    es_config = EarlyStoppingConfig(
            es_type="patient_converged",
            patience=20,
            n_check=20,
            divergence=0.01,
            burnin=2000,
            )
    loss_config = LossFunctionConfig(loss_type="mse_step")
    ds_splitter_config = DatasetSplitterConfig(split_type="temporal")


    logging.info(f"Optimizer config: {optimizer_config}")
    logging.info(f"GAT model config: {gat_model_config}")
    logging.info(f"Early stopping config: {es_config}")
    logging.info(f"Loss function config: {loss_config}")
    logging.info(f"Dataset splitter config: {ds_splitter_config}")


    gat_ds_config = DatasetConfig.from_exp_file(
        Path(exp_data_json),
    )

    t_gat = Trainer(
            optimizer_config=optimizer_config,
            model_config=gat_model_config,
            es_config=es_config,
            ds_config=gat_ds_config,
            ds_splitter_config=ds_splitter_config,
            loss_config=loss_config,
            n_epochs=5000,
            device="cuda",
            output_dir=output_dir,
            use_wandb=False,
    )
    t_gat.initialize()
    t_gat.train()



def _gather_and_clean_data(protocol_name: str):
    n_radicals = 0
    n_covalents = 0
    filtered_cdd_data_this_protocol = []
    cdd_data_this_protocol = download_cdd_data(protocol_name=protocol_name)

    for _, row in cdd_data_this_protocol.iterrows():
        logger.debug(f"Working on {row['Molecule Name']}..")
        # do checks first, based on https://github.com/choderalab/asapdiscovery/blob/main/asapdiscovery-alchemy/asapdiscovery/alchemy/cli/utils.py#L132-L146
        mol = Ligand.from_smiles(
            smiles=row["Smiles"],
            compound_name=row["Molecule Name"],
            cxsmiles=row["CXSmiles"],
        )

        # first remove compounds with radicals
        try:
            _ = Molecule.from_smiles(
                mol.smiles,
                allow_undefined_stereo=True,  # better for GAT as observed by BK's testing
            )
        except RadicalsNotSupportedError:
            n_radicals += 1
            logger.debug("Rejected because this compound contains a radical.")
            continue

        # then remove compounds with covalent warheads
        if has_warhead(ligand=mol):
            n_covalents += 1
            logger.debug("Rejected because this compound is a covalent binder.")
            continue

        # compound is safe to be added to MLOps training set for this protocol.
        logger.debug("Compound accepted.")
        filtered_cdd_data_this_protocol.append(row)
    logging.info(f"Rejected {n_radicals} compounds with radicals.")
    logging.info(f"Rejected {n_covalents} compounds with covalent warheads.")
    logging.info(f"Accepted {len(filtered_cdd_data_this_protocol)} compounds for training.")

    # now also apply BK's set of filters and wrangle into data type ready to be trained on.
    # based on https://asapdiscovery.readthedocs.io/en/latest/tutorials/training_ml_models_on_asap_data.html.
    this_protocol_training_set = parse_fluorescence_data_cdd(
    filter_molecules_dataframe(
        pd.DataFrame(filtered_cdd_data_this_protocol),
        id_fieldname="Molecule Name",
        smiles_fieldname="Smiles",
        assay_name=protocol_name,
        retain_achiral=True,
        retain_racemic=True,
        retain_enantiopure=True,
        retain_semiquantitative_data=True,
    ),
    assay_name=protocol_name,
    )
    return this_protocol_training_set


def _write_ensemble_manifest_yaml(model_tag, ensemble_directories, output_dir):
    ensemble_manifest = {
        "model_tag": model_tag,
        "ensemble": [
            {
                "model_tag": f"{model_tag}_ensemble_{i}",
                "weights": f"{model_tag}_ensemble_{i}.pth",
                "config": f"{model_tag}_ensemble_{i}.json",
            }
            for i in range(len(ensemble_directories))
        ],
    }

    manifest_path = output_dir / f"{model_tag}_ensemble_manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(ensemble_manifest, f)
    logging.info(f"Written ensemble manifest to {manifest_path}")
    return manifest_path



@click.group()
def mlops():
    pass


@mlops.command()
@click.option("-p", "--protocol", type=str, required=True, help="Endpoint to train GAT model for")
@output_dir
@loglevel
@click.option("-e", "--ensemble-size", type=int, default=5, help="Number of models in ensemble")
def train_GAT_for_endpoint(
    protocol: str,
    output_dir: str = "output",
    loglevel: str = "INFO",
    ensemble_size: int = 5,

):
    """
    Train a GAT model for a specific endpoint
    """
    if output_dir is None:
        output_dir = Path.cwd() / "output"
    

    # make the output directory, overwriting
    if output_dir.exists():
        rmtree(output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)


    logger = FileLogger("", path=output_dir, logfile="train_GAT_for_endpoint.log", level=loglevel, stdout=True).getLogger()

    ISO_TODAY = datetime.datetime.now().strftime("%Y-%m-%d")
    model_tag = f"asapdiscovery-GAT-{protocol}-ensemble-{ISO_TODAY}"


    logger.info(f"Training GAT model for endpoint {protocol}")
    logger.info(f"Start time: {ISO_TODAY}")
    logger.info(f"Model tag: {model_tag}")
    logger.info(f"Output directory: {output_dir}")

    # do some pre checking before we start
    try:
        s3_settings = S3Settings()
    except Exception as e:
        raise ValueError(f"Could not load S3 settings: {e}, quitting.")

    if protocol not in PROTOCOLS:
        raise ValueError(f"Endpoint {protocol} not in allowed list of protocols {PROTOCOLS}")



    # download the data for the endpoint
    this_protocol_training_set = _gather_and_clean_data(protocol)
    # cludge to set the date to the right hardcoded column values
    this_protocol_training_set.rename(columns={"modified_at": "Batch Created Date"}, inplace=True)

    # save the data
    out_csv = output_dir / f"{protocol}_training_set_{ISO_TODAY}.csv"
    this_protocol_training_set.to_csv(out_csv)
    logger.info(f"Saved training set to {out_csv}")


    # make output directory for this protocol
    protocol_out_dir = output_dir / protocol
    protocol_out_dir.mkdir()
    out_json = protocol_out_dir / f"{ISO_TODAY}_ml_gat_input.json"
    _ = cdd_to_schema(
    cdd_csv=out_csv,
    out_json=out_json)

    logger.info(f"Saved input JSON for GAT model training to {out_json}")

    # train the model
    logger.info(f"Training ensemble of {ensemble_size} models")


    ensemble_directories = []
    for i in range(ensemble_size):
        logger.info(f"Training ensemble model {i}")
        ensemble_out_dir = protocol_out_dir / f"ensemble_{i}"
        ensemble_out_dir.mkdir()
        _train_single_model(out_json, ensemble_out_dir)
        ensemble_directories.append(ensemble_out_dir)


    logger.info(f"Training complete for {protocol}")


    logger.info("writing ensemble manifest")

    manifset_path = _write_ensemble_manifest_yaml(model_tag, ensemble_directories, output_dir)

    # now push weights, config and manifest to S3
    logger.info("Pushing weights, config and manifest to S3")





    logger.info("Done.")




    



    
  



