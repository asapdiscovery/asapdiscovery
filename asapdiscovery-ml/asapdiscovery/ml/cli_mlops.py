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
from shutil import rmtree, copy
import os
import mtenn
import hashlib
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
    n_epochs
)
import torch
from hashlib import sha256
from asapdiscovery.data.services.services_config import S3Settings
from asapdiscovery.data.services.aws.s3 import S3
from asapdiscovery.cli.cli_args import loglevel

from openff.toolkit import Molecule
from openff.toolkit.utils.exceptions import (
    RadicalsNotSupportedError,
)
from asapdiscovery.ml.pretrained_models import cdd_protocols_yaml
import yaml
import logging
import datetime
import pandas as pd
from pathlib import Path
import yaml
import wandb



# logging 
logger = logging.getLogger(__name__)

"""
TODO:
- hook up to WandB and add hyperparameter search per model in ensemble ('sweep')
"""

PROTOCOLS = yaml.safe_load(open(cdd_protocols_yaml))["protocols"]


# need Py3.11 + for hashlib.file_digest, use this for now
def sha256sum(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, 'rb') as file:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = file.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()



def _train_single_model(ensemble_tag, model_tag, exp_data_json, output_dir, n_epochs=5000, wandb_project=None):

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


    logging.debug(f"Optimizer config: {optimizer_config}")
    logging.debug(f"GAT model config: {gat_model_config}")
    logging.debug(f"Early stopping config: {es_config}")
    logging.debug(f"Loss function config: {loss_config}")
    logging.debug(f"Dataset splitter config: {ds_splitter_config}")


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
            n_epochs=n_epochs,
            device="cuda" if torch.cuda.is_available() else "cpu",
            output_dir=output_dir,
            use_wandb=True,
            wandb_project=wandb_project,
            wandb_name=ensemble_tag,
            wandb_group=model_tag,
            save_weights="final"
    )
    t_gat.initialize()
    t_gat.train()
    # need to get dir for WANDB, as has run_id prefix
    return t_gat.output_dir



def _gather_and_clean_data(protocol_name: str, output_dir: Path = None):

    from asapdiscovery.data.services.cdd.cdd_api import CDDAPI
    from asapdiscovery.data.services.services_config import CDDSettings
    from asapdiscovery.data.util.utils import parse_fluorescence_data_cdd
    

    if protocol_name not in PROTOCOLS.keys():
        raise ValueError(f"Protocol {protocol_name} not in allowed list of protocols {PROTOCOLS}")


    settings = CDDSettings()
    cdd_api = CDDAPI.from_settings(settings=settings)

    readout = PROTOCOLS[protocol_name]
    if not readout:
        raise ValueError(f"readout type not found for {protocol_name}")

    if readout == "affinity":
        logging.debug(f"Getting IC50 data for {protocol_name}")
        ic50_data = cdd_api.get_ic50_data(protocol_name=protocol_name)
        # format the data to add the pIC50 and error
        cdd_data_this_protocol = parse_fluorescence_data_cdd(
            mol_df=ic50_data, assay_name=protocol_name
        )
    else:
        logging.debug(f"Getting readout data for {protocol_name} with readout {readout}")
        cdd_data_this_protocol = cdd_api.get_readout(protocol_name=protocol_name, readout=readout)


    n_radicals = 0
    n_covalents = 0
    filtered_cdd_data_this_protocol = []
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


    df =  pd.DataFrame(filtered_cdd_data_this_protocol)

    # kludge to set the date to the right hardcoded column values
    df.rename(columns={"modified_at": "Batch Created Date"}, inplace=True)
    df.to_csv(output_dir / "raw_filtered_cdd_data.csv")

    if readout == "activity":
        logger.info("Protocol is an activity fluoresence endpoint, parsing data accordingly")
        this_protocol_training_set = parse_fluorescence_data_cdd(
        filter_molecules_dataframe(
            df,
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
    else:
        logger.info("Protocol is a scalar endpoint, parsing data accordingly")
        this_protocol_training_set = filter_molecules_dataframe(
            df,
            id_fieldname="Molecule Name",
            smiles_fieldname="Smiles",
            assay_name=readout, # NB: this is the readout, not the protocol name, as we used get_readout
            retain_achiral=True,
            retain_racemic=True,
            retain_enantiopure=True,
            retain_semiquantitative_data=True,
            is_ic50=False, # need to add point to skip IC50 protocol conversion
        )
    
    return this_protocol_training_set



def _write_ensemble_manifest_yaml(model_tag, weights_paths, config_path, output_dir, protocol, ISO_TODAY):
    """
    Writes a YAML manifest for the ensemble of models trained for a specific endpoint

    Manifest looks like 


asapdiscovery-GAT-ensemble-test:
  type: GAT
  base_url: https://asap-discovery-ml-skynet.asapdata.org/test_manifest/endpoint/
  ensemble: True

  weights:
    - member1:
        resource: member1.th
        sha256hash: 4a6494412089d390723b107a30361672f2d2711622eea016c33caf1d7c28e1a7
    ...

  config:
    resource: config.json
    sha256hash: d26f278c189eb897607b9b3a2c61ed6c82fbcd7590683631dc9afd7fa010f256
  targets:
    - SARS-CoV-2-Mpro
    - MERS-CoV-Mpro
  mtenn_lower_pin: "0.5.0"
  last_updated: 2024-01-01
    """
    manifest = {}
    ensemble_manifest = {
        "type": "GAT",
        "base_url": f"https://asap-discovery-ml-skynet.asapdata.org/{protocol}/{model_tag}/",
        "ensemble": True,
        "weights": {},
        "config": {
            "resource": "model_config.json",
            "sha256hash": sha256sum(config_path)
        },
        "targets": [_protocol_to_target(protocol)],
        "mtenn_lower_pin": mtenn.__version__,
        "last_updated": ISO_TODAY
    }


    ensemble_manifest["weights"] = [{member: {"resource": weights_path.name, "sha256hash": sha256sum(weights_path)}} for member, weights_path in weights_paths.items()]
    manifest[model_tag] = ensemble_manifest
    manifest_path = output_dir / f"{model_tag}_manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f)
    return manifest_path


def _protocol_to_target(protocol):
    """
    Converts a protocol name to a list of targets
    """
    # grab the first element at underscore split
    target = protocol.split("_")[0]

    # normalize MPro to Mpro
    if "MPro" in target:
        target = target.replace("MPro", "Mpro")
    return target

   

def _gather_weights(ensemble_directories, model_tag, output_dir, ISO_TODAY):
    """
    Gathers the weights and config files from the ensemble directories and writes them to a final directory
    """
    final_dir = output_dir / model_tag
    final_dir.mkdir()
    weights_paths = {}
    for i, ensemble_dir in enumerate(ensemble_directories):
        member = f"member{i}"
        ens_weights_path = ensemble_dir / "final.th"
        # copy the weights to the final directory
        final_weights_path = final_dir / f"{member}.th"
        copy(ens_weights_path, final_weights_path)
        weights_paths[member] = final_weights_path
    
    config_path = ensemble_directories[0] / "model_config.json"
    # copy the config to the final directory
    final_config_path = final_dir / "model_config.json"
    copy(config_path, final_config_path)
    return final_dir, weights_paths, final_config_path



@click.group()
def mlops():
    pass


@mlops.command()
@click.option("-p", "--protocol", type=str, required=True, help="Endpoint to train GAT model for")
@output_dir
@loglevel
@click.option("-e", "--ensemble-size", type=int, default=5, help="Number of models in ensemble")
@click.option("-n", "--n-epochs", type=int, default=5000, help="Number of epochs to train for")
def train_GAT_for_endpoint(
    protocol: str,
    output_dir: str = "output",
    loglevel: str = "INFO",
    ensemble_size: int = 5,
    n_epochs: int = 5000,
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
        
    wandb_project = os.getenv("WANDB_PROJECT")

    if wandb_project is None:
        raise ValueError("WandB project not set, quitting.")

    if protocol not in PROTOCOLS.keys():
        raise ValueError(f"Endpoint {protocol} not in allowed list of protocols {PROTOCOLS}")


    # download the data for the endpoint
    this_protocol_training_set = _gather_and_clean_data(protocol, output_dir)

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
        ensemble_tag = f"{model_tag}_ensemble_{i}"
        logger.info(f"Training ensemble model {i}")
        ensemble_out_dir = protocol_out_dir / f"ensemble_{i}"
        ensemble_out_dir.mkdir()
        output_model_dir = _train_single_model(ensemble_tag, model_tag, out_json, ensemble_out_dir, wandb_project=wandb_project, n_epochs=n_epochs)
        ensemble_directories.append(output_model_dir)


    logger.info(f"Training complete for {protocol}")

    final_dir_path, weights_paths, config_path = _gather_weights(ensemble_directories, model_tag, output_dir, ISO_TODAY)

    logger.info(f"Final ensemble weights and config saved to {final_dir_path}")
    logger.info(f"weights_paths: {weights_paths}")
    logger.info(f"config_path: {config_path}")


    logger.info("writing ensemble manifest")

    manifest_path = _write_ensemble_manifest_yaml(model_tag, weights_paths, config_path, output_dir, protocol, ISO_TODAY)

    logger.info(f"Manifest written to {manifest_path}")

    # copy manifest to final directory
    final_manifest_path = final_dir_path / manifest_path.name
    copy(manifest_path, final_manifest_path)

    # now push weights, config and manifest to S3
    logger.info("Pushing weights, config and manifest to S3")
    # push the whole final directory to S3
    # ends up at BUCKET_NAME/protocol/model_tag
    s3 = S3.from_settings(s3_settings)
    s3_ensemble_dest = f"{protocol}/{model_tag}"

    # push ensemble to "latest"
    s3_manifest_dest = f"{protocol}/latest/manifest.yaml"


    logger.info(f"Pushing final directory to S3 at {s3_ensemble_dest}")
    s3.push_dir(final_dir_path, s3_ensemble_dest)

    logger.info(f"Pushing manifest to S3 at {s3_manifest_dest}")
    s3.push_file(manifest_path, s3_manifest_dest)

    logger.info("Done.")
