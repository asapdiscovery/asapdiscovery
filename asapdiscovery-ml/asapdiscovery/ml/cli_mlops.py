import datetime
import hashlib
import logging
import os
import shutil
from pathlib import Path
from shutil import copy, rmtree
from typing import Optional

import click
import matplotlib.pyplot as plt
import mtenn
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import torch
import wandb
import yaml
from asapdiscovery.alchemy.cli.utils import has_warhead
from asapdiscovery.cli.cli_args import loglevel
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.services.aws.s3 import S3
from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags
from asapdiscovery.data.services.services_config import S3Settings
from asapdiscovery.data.util.logging import FileLogger
from asapdiscovery.data.util.utils import (
    cdd_to_schema,
    cdd_to_schema_v2,
    filter_molecules_dataframe,
)
from asapdiscovery.ml.cli_args import output_dir
from asapdiscovery.ml.config import (
    DatasetConfig,
    DatasetSplitterConfig,
    EarlyStoppingConfig,
    LossFunctionConfig,
    OptimizerConfig,
)
from asapdiscovery.ml.inference import GATInference
from asapdiscovery.ml.models import RemoteEnsembleHelper
from asapdiscovery.ml.pretrained_models import cdd_protocols_yaml
from asapdiscovery.ml.trainer import Trainer
from mtenn.config import GATModelConfig
from openff.toolkit import Molecule
from openff.toolkit.utils.exceptions import RadicalsNotSupportedError

# logging
logger = logging.getLogger(__name__)


PROTOCOLS = yaml.safe_load(open(cdd_protocols_yaml))["protocols"]

SKYNET_SERVE_URL = "https://asap-discovery-ml-skynet.asapdata.org"


def evaluate_test_performance(
    test_csv, readout_column, model, output_dir, exp_err_column=None
):
    """
    Evaluate the performance of a trained model on a test set

    Parameters
    ----------
    test_csv : str
        Path to the test set CSV
    readout_column : str
        Column in the CSV containing the readout
    model : GATModelSpec
        Trained model
    output_dir : str
        Output directory for the plot
    exp_err_column : str
        Column in the CSV containing the experimental error

    Returns
    -------
    str
        Path to the plot
    """
    df = pd.read_csv(test_csv)
    inference_cls = GATInference.from_ml_model_spec(model)
    smiles = df["smiles"]
    pred, err = inference_cls.predict_from_smiles(smiles, return_err=True)
    pred_column = f"predicted_{readout_column}"
    err_column = f"prediction_{readout_column}err"
    df[pred_column] = pred
    df[err_column] = err
    out_plot = plot_test_performance(
        df,
        model.name,
        readout_column,
        pred_column,
        output_dir,
        err_column,
        exp_err_column=exp_err_column,
    )
    return out_plot


def plot_test_performance(
    df,
    model_name,
    readout_column,
    pred_column,
    output_dir,
    err_column,
    exp_err_column=None,
    plotname="test_performance.png",
) -> str:
    """
    Plot the performance of a trained model on a test set

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the test set
    model_name : str
        Name of the model
    readout_column : str
        Column in the DataFrame containing the readout
    pred_column : str
        Column in the DataFrame containing the predicted readout
    output_dir : str
        Output directory for the plot
    err_column : str
        Column in the DataFrame containing the error
    exp_err_column : str
        Column in the DataFrame containing the experimental error
    plotname : str
        Name of the plot

    Returns
    -------
    str
        Path to the plot
    """
    fig, ax = plt.subplots()
    ax.set_title(f"Test set performance:\n {model_name}", fontsize=6)
    min_val = min(df[readout_column].min(), df[pred_column].min())
    max_val = max(df[readout_column].max(), df[pred_column].max())
    # set the limits to be the same for both axes
    p = sns.regplot(x=readout_column, data=df, y=pred_column, ax=ax, ci=None)
    slope, intercept, r, p, sterr = scipy.stats.linregress(
        x=p.get_lines()[0].get_xdata(), y=p.get_lines()[0].get_ydata()
    )
    ax.set_aspect("equal", "box")
    min_ax = min_val - 1
    max_ax = max_val + 1

    ax.set_xlim(min_ax, max_ax)
    ax.set_ylim(min_ax, max_ax)
    # plot y = x line in dashed grey
    ax.plot([min_ax, max_ax], [min_ax, max_ax], linestyle="--", color="black")
    # Shade 0.5 and 1 unit regions around the y=x line
    ax.fill_between(
        [min_ax, max_ax],
        [min_ax - 0.5, max_ax - 0.5],
        [min_ax + 0.5, max_ax + 0.5],
        color="gray",
        alpha=0.2,
    )
    ax.fill_between(
        [min_ax, max_ax],
        [min_ax - 1, max_ax - 1],
        [min_ax + 1, max_ax + 1],
        color="gray",
        alpha=0.2,
    )
    # plot error bars
    ax.errorbar(
        df[readout_column],
        df[pred_column],
        yerr=df[err_column],
        fmt="none",
        capsize=5,
        zorder=1,
        color="C0",
    )
    if exp_err_column:
        ax.errorbar(
            df[readout_column],
            df[pred_column],
            xerr=df[exp_err_column],
            fmt="none",
            capsize=5,
            zorder=1,
            color="C0",
        )
    stats_dict = do_stats(df[readout_column], df[pred_column])
    stats_text = stats_to_str(stats_dict)
    ax.text(0.05, 0.8, stats_text, transform=ax.transAxes, fontsize=8)
    ax.text(
        0.05,
        0.75,
        f"y = {str(round(slope, 3))}x + {str(round(intercept, 3))}",
        transform=ax.transAxes,
        fontsize=8,
    )

    out = output_dir / plotname
    fig.tight_layout()
    plt.savefig(out)
    return out


def do_stats(target_vals, preds):
    from scipy.stats import bootstrap, kendalltau, spearmanr

    # Calculate MAE and bootstrapped confidence interval
    stats_dict = {}
    mae = np.abs(target_vals - preds).mean()
    conf_interval = bootstrap(
        (target_vals, preds),
        statistic=lambda target, pred: np.abs(target - pred).mean(),
        method="basic",
        confidence_level=0.95,
        paired=True,
    ).confidence_interval
    stats_dict["mae"] = {
        "value": mae,
        "95ci_low": conf_interval.low,
        "95ci_high": conf_interval.high,
    }

    # Calculate RMSE and bootstrapped confidence interval
    rmse = np.sqrt(np.power(target_vals - preds, 2).mean())
    conf_interval = bootstrap(
        (target_vals, preds),
        statistic=lambda target, pred: np.sqrt(np.power(target - pred, 2).mean()),
        method="basic",
        confidence_level=0.95,
        paired=True,
    ).confidence_interval
    stats_dict["rmse"] = {
        "value": rmse,
        "95ci_low": conf_interval.low,
        "95ci_high": conf_interval.high,
    }

    # Calculate Spearman r and bootstrapped confidence interval
    sp_r = spearmanr(target_vals, preds).statistic
    conf_interval = bootstrap(
        (target_vals, preds),
        statistic=lambda target, pred: spearmanr(target, pred).statistic,
        method="basic",
        confidence_level=0.95,
        paired=True,
    ).confidence_interval
    stats_dict["sp_r"] = {
        "value": sp_r,
        "95ci_low": conf_interval.low,
        "95ci_high": conf_interval.high,
    }

    # Calculate Kendall's tau and bootstrapped confidence interval
    tau = kendalltau(target_vals, preds).statistic
    conf_interval = bootstrap(
        (target_vals, preds),
        statistic=lambda target, pred: kendalltau(target, pred).statistic,
        method="basic",
        confidence_level=0.95,
        paired=True,
    ).confidence_interval
    stats_dict["tau"] = {
        "value": tau,
        "95ci_low": conf_interval.low,
        "95ci_high": conf_interval.high,
    }

    return stats_dict


def stats_to_str(stats_dict):
    stats_text = []
    for stat, stat_label in zip(
        ["mae", "rmse", "sp_r", "tau"],
        ["MAE", "RMSE", "Spearman's $\\rho$", "Kendall's $\\tau$"],
    ):
        stats_str = (
            f"{stat_label}: "
            f"{stats_dict[stat]['value']:0.2f}"
            f"$_{{{stats_dict[stat]['95ci_low']:0.2f}}}"
            f"^{{{stats_dict[stat]['95ci_high']:0.2f}}}$"
        )
        stats_text.append(stats_str)
    return "\n".join(stats_text)


# need Py3.11 + for hashlib.file_digest, use this for now
def sha256sum(file_path: Path) -> str:
    """
    Calculate the SHA256 hash of a file

    Parameters
    ----------
    file_path : Path
        Path to the file

    Returns
    -------
    str
        SHA256 hash of the file
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as file:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = file.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _train_single_model(
    target_prop: str,
    ensemble_tag: str,
    model_tag: str,
    exp_data_json: Path,
    output_dir: Path,
    n_epochs: int = 5000,
    wandb_project: Optional[str] = None,
):
    """
    Train a single GAT model for a specific endpoint

    Parameters
    ----------
    target_prop : str
        Target property to train the model for, must be specified directly as a readout from CDD or "pIC50" which uses the uncertainty and range
    ensemble_tag : str
        Tag of this model in the ensemble
    model_tag : str
        Tag of the ensemble
    exp_data_json : str
        Path to the JSON file containing the experimental data pulled from CDD
    output_dir : Path
        Output directory for the trained model
    n_epochs : int
        Number of epochs to train for
    wandb_project : str
        WandB project to log to

    Returns
    -------
    Path
        Path to the output directory of the trained model, as set by wandb
    """

    logging.info(
        f'Training GAT model for {exp_data_json} with target property "{target_prop}"'
    )
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

    # pIC0s have uncertainty and range, scalar endpoints have neither
    if target_prop == "pIC50":
        has_uncertainty = True
        has_range = True
    else:
        has_uncertainty = False
        has_range = False

    logger.info(
        f"Training GAT model for {target_prop} with uncertainty={has_uncertainty} and range={has_range}"
    )

    t_gat = Trainer(
        target_prop=target_prop,
        optimizer_config=optimizer_config,
        model_config=gat_model_config,
        es_config=es_config,
        ds_config=gat_ds_config,
        ds_splitter_config=ds_splitter_config,
        loss_configs=[loss_config],
        n_epochs=n_epochs,
        device="cuda" if torch.cuda.is_available() else "cpu",  # let PyTorch decide
        output_dir=output_dir,
        use_wandb=True,
        wandb_project=wandb_project,
        wandb_name=ensemble_tag,
        wandb_group=model_tag,
        save_weights="final",
        has_uncertainty=has_uncertainty,
        has_range=has_range,
        write_ds_csv=True,
    )
    t_gat.initialize()
    t_gat.train()
    # need to get dir for output as set by W&B inside trainer, as has run_id prefix
    return t_gat.output_dir, t_gat.wandb_run_id


def _gather_and_clean_data(protocol_name: str, output_dir: Path = None) -> pd.DataFrame:
    """
    Gather and clean data for a specific endpoint from CDD. Handles special cases for pIC50 data
    or scalar endpoints, for which the readout is directly used. The data is cleaned to remove radicals and covalent warheads.

    Parameters
    ----------
    protocol_name : str
        Name of the protocol to gather data for
    output_dir : Path
        Output directory to save the raw data to

    Returns
    -------
    pd.DataFrame
        DataFrame containing the cleaned data
    """

    from asapdiscovery.data.services.cdd.cdd_api import CDDAPI
    from asapdiscovery.data.services.services_config import CDDSettings
    from asapdiscovery.data.util.utils import parse_fluorescence_data_cdd

    if protocol_name not in PROTOCOLS.keys():
        raise ValueError(
            f"Protocol {protocol_name} not in allowed list of protocols {PROTOCOLS}"
        )
    readout = PROTOCOLS[protocol_name]["readout"]
    if not readout:
        raise ValueError(f"readout type not found for {protocol_name}")

    target = _protocol_to_target(protocol_name)

    if target is None:
        logger.info(
            "Target-less protocol"
        )  # some protocols don't have a target, e.g logD
    else:
        logger.info(f"Target for protocol {protocol_name} is {target}")

    endpoint = _protocol_to_endpoint(protocol_name)
    if endpoint is None:
        raise ValueError(f"Endpoint not found for protocol {protocol_name}")
    else:
        logger.info(f"Endpoint for protocol {protocol_name} is {endpoint}")

    # if its a string though, we need to check it is in allowed list of targets
    if isinstance(target, str):
        if target not in TargetTags.get_values():
            raise ValueError(
                f"Target {target} not in allowed list of targets {TargetTags.get_values()}"
            )

    try:
        settings = CDDSettings()
        cdd_api = CDDAPI.from_settings(settings=settings)
    except Exception as e:
        raise ValueError(f"Could not load CDD settings: {e}, quitting.")

    if readout == "pIC50":
        # special case for PIC50 data, which uses several readouts, including errs etc
        logging.debug(f"Getting IC50 data for {protocol_name}")
        ic50_data = cdd_api.get_ic50_data(protocol_name=protocol_name)
        # format the data to add the pIC50 and error
        cdd_data_this_protocol = parse_fluorescence_data_cdd(
            mol_df=ic50_data, assay_name=protocol_name
        )
        # drop values where pIC50 rounds to <= 0 or >= 10, caused by massive error bars.
        # TODO should be fixed upstream in future #1234
        cdd_data_this_protocol = cdd_data_this_protocol[
            (cdd_data_this_protocol["pIC50"] > 0)
            & (cdd_data_this_protocol["pIC50"] < 10)
        ]
        # log which compounds were dropped
        dropped = ic50_data[
            ~ic50_data["Molecule Name"].isin(cdd_data_this_protocol["Molecule Name"])
        ]
        logging.info(f"Dropped {len(dropped)} compounds with pIC50 <= 0 or >= 10")
        for _, row in dropped.iterrows():
            logging.debug(f"Compound {row['Molecule Name']} dropped.")

    else:
        # otherwise we pull the readout directly
        logging.debug(
            f"Getting readout data for {protocol_name} with readout {readout}"
        )
        cdd_data_this_protocol = cdd_api.get_readout(
            protocol_name=protocol_name, readout=readout
        )

    # do some pre-processing to remove compounds with radicals or covalent warheads

    n_radicals = 0
    n_covalents = 0
    filtered_cdd_data_this_protocol = []
    for _, row in cdd_data_this_protocol.iterrows():
        logger.debug(f"Working on {row['Molecule Name']}..")
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
    logging.info(
        f"Accepted {len(filtered_cdd_data_this_protocol)} compounds for training."
    )

    df = pd.DataFrame(filtered_cdd_data_this_protocol)

    # kludge to set the date to the right hardcoded column values
    df.rename(columns={"modified_at": "Batch Created Date"}, inplace=True)

    df.to_csv(output_dir / "raw_filtered_cdd_data.csv")

    if readout == "pIC50":
        logger.info("Protocol is an IC50 activity endpoint, parsing data accordingly")
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
            assay_name=readout,  # NB: this is the readout, not the protocol name, as we used get_readout
            retain_achiral=True,
            retain_racemic=True,
            retain_enantiopure=True,
            retain_semiquantitative_data=True,
            is_ic50=False,  # need to add point to skip IC50 protocol conversion #TODO: refactor this so that base assumption is not IC50
        )

    return this_protocol_training_set


def _write_ensemble_manifest_yaml(
    model_tag: str,
    weights_paths: dict[str, Path],
    config_path: Path,
    output_dir: Path,
    protocol: str,
    ISO_TODAY: datetime.datetime,
) -> Path:
    """
        Writes a YAML manifest for the ensemble of models trained for a specific endpoint

        Manifest looks like:


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

    Parameters
    ----------
    model_tag : str
        Tag of the ensemble
    weights_paths : dict[str, Path]
        Dictionary of model weights paths
    config_path : Path
        Path to the model config file
    output_dir : Path
        Output directory for the manifest
    protocol : str
        Name of the protocol
    ISO_TODAY : datetime.datetime
        Current date in ISO format

    Returns
    -------
    Path
        Path to the manifest file
    """
    manifest = {}
    ensemble_manifest = {
        "type": "GAT",
        "base_url": f"{SKYNET_SERVE_URL}/{protocol}/{model_tag}/",
        "ensemble": True,
        "weights": {},
        "config": {
            "resource": "model_config.json",
            "sha256hash": sha256sum(config_path),
        },
        "targets": [_protocol_to_target(protocol)],
        "endpoint": _protocol_to_endpoint(protocol),
        "mtenn_lower_pin": mtenn.__version__,
        "last_updated": ISO_TODAY,
    }

    ensemble_manifest["weights"] = [
        {member: {"resource": weights_path.name, "sha256hash": sha256sum(weights_path)}}
        for member, weights_path in weights_paths.items()
    ]
    manifest[model_tag] = ensemble_manifest
    manifest_path = output_dir / f"{model_tag}_manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f)
    return manifest_path


def _protocol_to_target(protocol: str) -> str:
    """
    Converts a protocol name to a target name

    Parameters
    ----------
    protocol : str
        Protocol name
    """
    try:
        target = PROTOCOLS[protocol]["target"]
    except KeyError:
        target = None
    return target


def _protocol_to_readout(protocol: str) -> str:
    """
    Converts a protocol name to a readout name

    Parameters
    ----------
    protocol : str
        Protocol name
    """
    try:
        readout = PROTOCOLS[protocol]["readout"]
    except KeyError:
        readout = None
    return readout


def _protocol_to_endpoint(protocol: str) -> str:
    """
    Converts a protocol name to an endpoint name

    Parameters
    ----------
    protocol : str
        Protocol name
    """
    try:
        endpoint = PROTOCOLS[protocol]["endpoint"]
    except KeyError:
        endpoint = None
    return endpoint


def _gather_weights(
    ensemble_directories: list[Path],
    model_tag: str,
    output_dir: Path,
    ISO_TODAY: datetime.datetime,
) -> tuple[Path, dict[str, Path], Path]:
    """
    Gathers the weights and config files from the ensemble directories and writes them to a final directory

    Parameters
    ----------
    ensemble_directories : list[Path]
        List of ensemble directories
    model_tag : str
        Tag of the ensemble
    output_dir : Path
        Output directory for the final ensemble
    ISO_TODAY : datetime.datetime
        Current date in ISO format

    Returns
    -------
    tuple[Path, dict[str, Path], Path]
        Path to the final directory, dictionary of weights paths, path to the config file
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
    test_csv_path = ensemble_directories[0] / "ds_test.csv"
    # copy the config to the final directory
    final_config_path = final_dir / "model_config.json"
    copy(config_path, final_config_path)
    return final_dir, weights_paths, final_config_path, test_csv_path


@click.group()
def mlops():
    pass


@mlops.command()
@click.option(
    "-p", "--protocol", type=str, required=True, help="Endpoint to train GAT model for"
)
@output_dir
@loglevel
@click.option(
    "-e", "--ensemble-size", type=int, default=5, help="Number of models in ensemble"
)
@click.option(
    "-n", "--n-epochs", type=int, default=5000, help="Number of epochs to train for"
)
@click.option(
    "-test",
    "--test",
    is_flag=True,
    help="Run in test mode, no S3 push or WandB logging to main project",
)
def train_GAT_for_endpoint(
    protocol: str,
    output_dir: str = "output",
    loglevel: str = "INFO",
    ensemble_size: int = 5,
    n_epochs: int = 5000,
    test: bool = False,
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

    logger = FileLogger(
        "",
        path=output_dir,
        logfile="train_GAT_for_endpoint.log",
        level=loglevel,
        stdout=True,
    ).getLogger()

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

    if test:
        logger.info("Test mode, not pushing to S3")

    wandb_project = os.getenv("WANDB_PROJECT")

    if wandb_project is None:
        raise ValueError("WandB project not set, quitting.")

    logger.info(f"WandB project: {wandb_project}")

    if protocol not in PROTOCOLS.keys():
        raise ValueError(
            f"Endpoint {protocol} not in allowed list of protocols {PROTOCOLS}"
        )

    readout = PROTOCOLS[protocol]["readout"]
    if not readout:
        raise ValueError(f"readout type not found for {protocol}")

    readout_err = PROTOCOLS[protocol]["readout_err"]

    logger.info(
        f'Endpoint "{protocol}" has readout: "{readout}" with error metric "{readout_err}", will be used as target property for training'
    )

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

    # pIC50 readouts have a bunch of munging in cdd_to_schema, so we need to handle them separately
    # this should really be refactored to be more general
    if readout == "pIC50":
        _ = cdd_to_schema(cdd_csv=out_csv, out_json=out_json)
    else:
        _ = cdd_to_schema_v2(
            target_prop=readout,
            time_column="Batch Created Date",
            cdd_csv=out_csv,
            out_json=out_json,
        )

    logger.info(f"Saved input JSON for GAT model training to {out_json}")

    # train the model
    logger.info(f"Training ensemble of {ensemble_size} models")

    # train each model in the ensemble
    ensemble_directories = []
    wandb_run_ids = []
    for i in range(ensemble_size):
        ensemble_tag = f"{model_tag}_ensemble_{i}"
        logger.info(f"Training ensemble model {i}")
        ensemble_out_dir = protocol_out_dir / f"ensemble_{i}"
        ensemble_out_dir.mkdir()
        output_model_dir, wandb_run_id = _train_single_model(
            readout,
            ensemble_tag,
            model_tag,
            out_json,
            ensemble_out_dir,
            wandb_project=wandb_project,
            n_epochs=n_epochs,
        )
        ensemble_directories.append(output_model_dir)
        wandb_run_ids.append(wandb_run_id)

    logger.info(f"Training complete for {protocol}")

    # gather the weights and config files
    final_dir_path, weights_paths, config_path, test_csv_path = _gather_weights(
        ensemble_directories, model_tag, output_dir, ISO_TODAY
    )

    logger.info(f"Final ensemble weights and config saved to {final_dir_path}")
    logger.info(f"weights_paths: {weights_paths}")
    logger.info(f"config_path: {config_path}")

    logger.info("writing ensemble manifest")

    # write the ensemble manifest
    manifest_path = _write_ensemble_manifest_yaml(
        model_tag, weights_paths, config_path, output_dir, protocol, ISO_TODAY
    )

    logger.info(f"Manifest written to {manifest_path}")

    # copy manifest to final directory
    final_manifest_path = final_dir_path / manifest_path.name
    copy(manifest_path, final_manifest_path)

    # need for both test and not test
    s3_manifest_dest = f"{protocol}/latest/manifest.yaml"

    if test:
        logger.info("Test mode, not pushing to S3")
    else:
        # now push weights, config and manifest to S3
        logger.info("Pushing weights, config and manifest to S3")
        # push the whole final directory to S3
        # ends up at BUCKET_NAME/protocol/model_tag
        s3 = S3.from_settings(s3_settings)
        s3_ensemble_dest = f"{protocol}/{model_tag}"

        # push ensemble to "latest"
        logger.info(f"Pushing final directory to S3 at {s3_ensemble_dest}")
        s3.push_dir(final_dir_path, s3_ensemble_dest)

        logger.info(f"Pushing manifest to S3 at {s3_manifest_dest}")
        s3.push_file(manifest_path, s3_manifest_dest)

    logger.info("Done pushing")
    # grab test CSV for plotting
    ds_test_end_path = output_dir / "ds_test.csv"
    shutil.copy(test_csv_path, ds_test_end_path)
    logger.info(f"Test CSV copied from  {test_csv_path} to {ds_test_end_path}")
    # use actual manifest URL for the next step, not just our push location, if testing will pull down older manifest
    manifest_url = f"{SKYNET_SERVE_URL}/{s3_manifest_dest}"
    logger.info(f"Manifest URL: {manifest_url}")
    # use the manifest we just made to make a new manifest for the latest ensemble
    logger.info("Plotting test performance for ensemble over test set")
    ens_models = RemoteEnsembleHelper(manifest_url=manifest_url).to_ensemble_spec()
    # dict with one item, grab the model
    model = list(ens_models.values())[0]
    logger.info(f"Model: {model}")
    plot_path = evaluate_test_performance(
        ds_test_end_path, readout, model, output_dir, exp_err_column=readout_err
    )
    logger.info(f"Test performance plot saved to {plot_path}")

    if test:
        logger.info("Test mode, not pushing to S3")
    else:
        # push the plot to S3 in the same location as the ensemble
        s3_plot_ens_dest = f"{protocol}/{model_tag}/test_performance.png"
        logger.info(f"Pushing test performance plot to S3 at {s3_plot_ens_dest}")
        s3.push_file(plot_path, s3_plot_ens_dest)

    # log the artifact to WandB run
    logger.info("Logging test performance plot to WandB")
    for run_id in wandb_run_ids:
        if run_id is None:
            continue
        try:
            wandb.init(id=run_id, project=wandb_project, resume="must")
            wandb.log({"test_performance_ensemble": wandb.Image(str(plot_path))})
            wandb.finish()
        except wandb.errors.error.UsageError as e:
            logger.error(f"WandB error: {e} for run {run_id}, failed to append plot")
