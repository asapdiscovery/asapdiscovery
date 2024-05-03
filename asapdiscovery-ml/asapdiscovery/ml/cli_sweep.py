import json
from pathlib import Path

import click
import pydantic
import torch
from asapdiscovery.data.util.utils import MOONSHOT_CDD_ID_REGEX, MPRO_ID_REGEX
from asapdiscovery.ml.cli_args import (
    ds_split_args,
    e3nn_args,
    es_args,
    force_new_sweep,
    gat_args,
    graph_ds_args,
    loss_args,
    model_config_cache,
    model_rand_seed,
    mtenn_args,
    optim_args,
    output_dir,
    overwrite_args,
    schnet_args,
    struct_ds_args,
    sweep_config,
    sweep_config_cache,
    sweep_config_cache_overwrite,
    trainer_args,
    trainer_config_cache,
    visnet_args,
    wandb_args,
    weights_path,
)
from asapdiscovery.ml.config import (
    DatasetSplitterType,
    EarlyStoppingType,
    LossFunctionType,
    OptimizerType,
)
from asapdiscovery.ml.sweep import Sweeper
from mtenn.config import CombinationConfig, ModelType, ReadoutConfig, StrategyConfig


@click.group()
def sweep():
    pass


@sweep.command(name="gat")
@output_dir
@weights_path
@trainer_config_cache
@sweep_config_cache
@optim_args
@wandb_args
@model_config_cache
@model_rand_seed
@mtenn_args
@gat_args
@es_args
@graph_ds_args
@ds_split_args
@loss_args
@trainer_args
@sweep_config
@force_new_sweep
@sweep_config_cache_overwrite
@overwrite_args
def sweep_gat(
    output_dir: Path | None = None,
    weights_path: Path | None = None,
    trainer_config_cache: Path | None = None,
    sweep_config_cache: Path | None = None,
    optimizer_type: OptimizerType | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    momentum: float | None = None,
    dampening: float | None = None,
    b1: float | None = None,
    b2: float | None = None,
    eps: float | None = None,
    rho: float | None = None,
    optimizer_config_cache: Path | None = None,
    use_wandb: bool | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    extra_config: list[str] | None = None,
    grouped: bool | None = None,
    strategy: StrategyConfig | None = None,
    pred_readout: ReadoutConfig | None = None,
    combination: CombinationConfig | None = None,
    comb_readout: ReadoutConfig | None = None,
    max_comb_neg: bool | None = None,
    max_comb_scale: float | None = None,
    pred_substrate: float | None = None,
    pred_km: float | None = None,
    comb_substrate: float | None = None,
    comb_km: float | None = None,
    model_config_cache: Path | None = None,
    model_rand_seed: int | None = None,
    in_feats: int | None = None,
    num_layers: int | None = None,
    hidden_feats: str | None = None,
    num_heads: str | None = None,
    feat_drops: str | None = None,
    attn_drops: str | None = None,
    alphas: str | None = None,
    residuals: str | None = None,
    agg_modes: str | None = None,
    biases: str | None = None,
    allow_zero_in_degree: bool | None = None,
    es_type: EarlyStoppingType | None = None,
    es_patience: int | None = None,
    es_n_check: int | None = None,
    es_divergence: float | None = None,
    es_config_cache: Path | None = None,
    exp_file: Path | None = None,
    ds_cache: Path | None = None,
    ds_config_cache: Path | None = None,
    ds_split_type: DatasetSplitterType | None = None,
    train_frac: float | None = None,
    val_frac: float | None = None,
    test_frac: float | None = None,
    enforce_one: bool | None = None,
    ds_rand_seed: int | None = None,
    ds_split_config_cache: Path | None = None,
    loss_type: LossFunctionType | None = None,
    semiquant_fill: float | None = None,
    loss_config_cache: Path | None = None,
    auto_init: bool | None = None,
    start_epoch: int | None = None,
    n_epochs: int | None = None,
    batch_size: int | None = None,
    target_prop: str | None = None,
    cont: bool | None = None,
    loss_dict: dict | None = None,
    device: torch.device | None = None,
    sweep_config: Path | None = None,
    force_new_sweep: bool | None = None,
    overwrite_sweep_config_cache: bool = False,
    overwrite_trainer_config_cache: bool = False,
    overwrite_optimizer_config_cache: bool = False,
    overwrite_model_config_cache: bool = False,
    overwrite_es_config_cache: bool = False,
    overwrite_ds_config_cache: bool = False,
    overwrite_ds_cache: bool = False,
    overwrite_ds_split_config_cache: bool = False,
    overwrite_loss_config_cache: bool = False,
):
    # Build each dict and pass to Trainer
    optim_config = {
        "cache": optimizer_config_cache,
        "overwrite_cache": overwrite_optimizer_config_cache,
        "optimizer_type": optimizer_type,
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "dampening": dampening,
        "b1": b1,
        "b2": b2,
        "eps": eps,
        "rho": rho,
    }
    model_config = {
        "cache": model_config_cache,
        "overwrite_cache": overwrite_model_config_cache,
        "model_type": ModelType.GAT,
        "rand_seed": model_rand_seed,
        "weights_path": weights_path,
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
        "num_layers": num_layers,
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
    es_config = {
        "cache": es_config_cache,
        "overwrite_cache": overwrite_es_config_cache,
        "es_type": es_type,
        "patience": es_patience,
        "n_check": es_n_check,
        "divergence": es_divergence,
    }
    ds_config = {
        "cache": ds_config_cache,
        "overwrite_cache": overwrite_ds_config_cache,
        "exp_file": exp_file,
        "is_structural": False,
        "cache_file": ds_cache,
        "overwrite": overwrite_ds_cache,
    }

    ds_splitter_config = {
        "cache": ds_split_config_cache,
        "overwrite_cache": overwrite_ds_split_config_cache,
        "split_type": ds_split_type,
        "grouped": grouped,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "enforce_one": enforce_one,
        "rand_seed": ds_rand_seed,
    }
    loss_config = {
        "cache": loss_config_cache,
        "overwrite_cache": overwrite_loss_config_cache,
        "loss_type": loss_type,
        "semiquant_fill": semiquant_fill,
    }

    # Parse loss_dict
    if loss_dict:
        loss_dict = json.loads(loss_dict.read_text())

    trainer_kwargs = {
        "optimizer_config": optim_config,
        "model_config": model_config,
        "es_config": es_config,
        "ds_config": ds_config,
        "ds_splitter_config": ds_splitter_config,
        "loss_config": loss_config,
        "auto_init": auto_init,
        "start_epoch": start_epoch,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "target_prop": target_prop,
        "cont": cont,
        "loss_dict": loss_dict,
        "device": device,
        "output_dir": output_dir,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "wandb_name": wandb_name,
        "extra_config": extra_config,
    }

    sweep_kwargs = {
        "sweep_config": sweep_config,
        "force_new_sweep": force_new_sweep,
    }

    sweeper = _build_sweeper(
        trainer_kwargs,
        sweep_kwargs,
        trainer_config_cache,
        sweep_config_cache,
        overwrite_sweep_config_cache,
    )

    sweeper.start_continue_sweep()


@sweep.command(name="schnet")
@output_dir
@weights_path
@trainer_config_cache
@sweep_config_cache
@optim_args
@model_config_cache
@model_rand_seed
@wandb_args
@mtenn_args
@schnet_args
@es_args
@graph_ds_args
@struct_ds_args
@ds_split_args
@loss_args
@trainer_args
@sweep_config
@force_new_sweep
@sweep_config_cache_overwrite
@overwrite_args
def sweep_schnet(
    output_dir: Path | None = None,
    weights_path: Path | None = None,
    trainer_config_cache: Path | None = None,
    sweep_config_cache: Path | None = None,
    optimizer_type: OptimizerType | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    momentum: float | None = None,
    dampening: float | None = None,
    b1: float | None = None,
    b2: float | None = None,
    eps: float | None = None,
    rho: float | None = None,
    optimizer_config_cache: Path | None = None,
    use_wandb: bool | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    extra_config: list[str] | None = None,
    grouped: bool | None = None,
    strategy: StrategyConfig | None = None,
    pred_readout: ReadoutConfig | None = None,
    combination: CombinationConfig | None = None,
    comb_readout: ReadoutConfig | None = None,
    max_comb_neg: bool | None = None,
    max_comb_scale: float | None = None,
    pred_substrate: float | None = None,
    pred_km: float | None = None,
    comb_substrate: float | None = None,
    comb_km: float | None = None,
    model_config_cache: Path | None = None,
    model_rand_seed: int | None = None,
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
    es_type: EarlyStoppingType | None = None,
    es_patience: int | None = None,
    es_n_check: int | None = None,
    es_divergence: float | None = None,
    es_config_cache: Path | None = None,
    exp_file: Path | None = None,
    ds_cache: Path | None = None,
    ds_config_cache: Path | None = None,
    structures: str | None = None,
    xtal_regex: str = MPRO_ID_REGEX,
    cpd_regex: str = MOONSHOT_CDD_ID_REGEX,
    ds_split_type: DatasetSplitterType | None = None,
    train_frac: float | None = None,
    val_frac: float | None = None,
    test_frac: float | None = None,
    enforce_one: bool | None = None,
    ds_rand_seed: int | None = None,
    ds_split_config_cache: Path | None = None,
    loss_type: LossFunctionType | None = None,
    semiquant_fill: float | None = None,
    loss_config_cache: Path | None = None,
    auto_init: bool | None = None,
    start_epoch: int | None = None,
    n_epochs: int | None = None,
    batch_size: int | None = None,
    target_prop: str | None = None,
    cont: bool | None = None,
    loss_dict: dict | None = None,
    device: torch.device | None = None,
    sweep_config: Path | None = None,
    force_new_sweep: bool | None = None,
    overwrite_sweep_config_cache: bool = False,
    overwrite_trainer_config_cache: bool = False,
    overwrite_optimizer_config_cache: bool = False,
    overwrite_model_config_cache: bool = False,
    overwrite_es_config_cache: bool = False,
    overwrite_ds_config_cache: bool = False,
    overwrite_ds_cache: bool = False,
    overwrite_ds_split_config_cache: bool = False,
    overwrite_loss_config_cache: bool = False,
):
    # Build each dict and pass to Trainer
    optim_config = {
        "cache": optimizer_config_cache,
        "overwrite_cache": overwrite_optimizer_config_cache,
        "optimizer_type": optimizer_type,
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "dampening": dampening,
        "b1": b1,
        "b2": b2,
        "eps": eps,
        "rho": rho,
    }
    model_config = {
        "cache": model_config_cache,
        "overwrite_cache": overwrite_model_config_cache,
        "model_type": ModelType.schnet,
        "rand_seed": model_rand_seed,
        "weights_path": weights_path,
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
    es_config = {
        "cache": es_config_cache,
        "overwrite_cache": overwrite_es_config_cache,
        "es_type": es_type,
        "patience": es_patience,
        "n_check": es_n_check,
        "divergence": es_divergence,
    }
    ds_config = {
        "cache": ds_config_cache,
        "overwrite_cache": overwrite_ds_config_cache,
        "exp_file": exp_file,
        "is_structural": True,
        "structures": structures,
        "xtal_regex": xtal_regex,
        "cpd_regex": cpd_regex,
        "cache_file": ds_cache,
        "overwrite": overwrite_ds_cache,
        "grouped": grouped,
        "for_e3nn": False,
    }

    ds_splitter_config = {
        "cache": ds_split_config_cache,
        "overwrite_cache": overwrite_ds_split_config_cache,
        "split_type": ds_split_type,
        "grouped": grouped,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "enforce_one": enforce_one,
        "rand_seed": ds_rand_seed,
    }
    loss_config = {
        "cache": loss_config_cache,
        "overwrite_cache": overwrite_loss_config_cache,
        "loss_type": loss_type,
        "semiquant_fill": semiquant_fill,
    }

    # Parse loss_dict
    if loss_dict:
        loss_dict = json.loads(loss_dict.read_text())

    trainer_kwargs = {
        "optimizer_config": optim_config,
        "model_config": model_config,
        "es_config": es_config,
        "ds_config": ds_config,
        "ds_splitter_config": ds_splitter_config,
        "loss_config": loss_config,
        "auto_init": auto_init,
        "start_epoch": start_epoch,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "target_prop": target_prop,
        "cont": cont,
        "loss_dict": loss_dict,
        "device": device,
        "output_dir": output_dir,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "wandb_name": wandb_name,
        "extra_config": extra_config,
    }

    sweep_kwargs = {
        "sweep_config": sweep_config,
        "force_new_sweep": force_new_sweep,
    }

    sweeper = _build_sweeper(
        trainer_kwargs,
        sweep_kwargs,
        trainer_config_cache,
        sweep_config_cache,
        overwrite_sweep_config_cache,
    )

    sweeper.start_continue_sweep()


@sweep.command("e3nn")
@output_dir
@weights_path
@trainer_config_cache
@sweep_config_cache
@optim_args
@model_config_cache
@model_rand_seed
@wandb_args
@mtenn_args
@e3nn_args
@es_args
@graph_ds_args
@struct_ds_args
@ds_split_args
@loss_args
@trainer_args
@sweep_config
@force_new_sweep
@sweep_config_cache_overwrite
@overwrite_args
def sweep_e3nn(
    output_dir: Path | None = None,
    weights_path: Path | None = None,
    trainer_config_cache: Path | None = None,
    sweep_config_cache: Path | None = None,
    optimizer_type: OptimizerType | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    momentum: float | None = None,
    dampening: float | None = None,
    b1: float | None = None,
    b2: float | None = None,
    eps: float | None = None,
    rho: float | None = None,
    optimizer_config_cache: Path | None = None,
    use_wandb: bool | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    extra_config: list[str] | None = None,
    grouped: bool | None = None,
    strategy: StrategyConfig | None = None,
    pred_readout: ReadoutConfig | None = None,
    combination: CombinationConfig | None = None,
    comb_readout: ReadoutConfig | None = None,
    max_comb_neg: bool | None = None,
    max_comb_scale: float | None = None,
    pred_substrate: float | None = None,
    pred_km: float | None = None,
    comb_substrate: float | None = None,
    comb_km: float | None = None,
    model_config_cache: Path | None = None,
    model_rand_seed: int | None = None,
    num_atom_types: int | None = None,
    irreps_hidden: str | None = None,
    lig: bool | None = None,
    irreps_edge_attr: int | None = None,
    num_layers: int | None = None,
    neighbor_dist: float | None = None,
    num_basis: int | None = None,
    num_radial_layers: int | None = None,
    num_radial_neurons: int | None = None,
    num_neighbors: float | None = None,
    num_nodes: float | None = None,
    es_type: EarlyStoppingType | None = None,
    es_patience: int | None = None,
    es_n_check: int | None = None,
    es_divergence: float | None = None,
    es_config_cache: Path | None = None,
    exp_file: Path | None = None,
    ds_cache: Path | None = None,
    ds_config_cache: Path | None = None,
    structures: str | None = None,
    xtal_regex: str = MPRO_ID_REGEX,
    cpd_regex: str = MOONSHOT_CDD_ID_REGEX,
    ds_split_type: DatasetSplitterType | None = None,
    train_frac: float | None = None,
    val_frac: float | None = None,
    test_frac: float | None = None,
    enforce_one: bool | None = None,
    ds_rand_seed: int | None = None,
    ds_split_config_cache: Path | None = None,
    loss_type: LossFunctionType | None = None,
    semiquant_fill: float | None = None,
    loss_config_cache: Path | None = None,
    auto_init: bool | None = None,
    start_epoch: int | None = None,
    n_epochs: int | None = None,
    batch_size: int | None = None,
    target_prop: str | None = None,
    cont: bool | None = None,
    loss_dict: dict | None = None,
    device: torch.device | None = None,
    sweep_config: Path | None = None,
    force_new_sweep: bool | None = None,
    overwrite_sweep_config_cache: bool = False,
    overwrite_trainer_config_cache: bool = False,
    overwrite_optimizer_config_cache: bool = False,
    overwrite_model_config_cache: bool = False,
    overwrite_es_config_cache: bool = False,
    overwrite_ds_config_cache: bool = False,
    overwrite_ds_cache: bool = False,
    overwrite_ds_split_config_cache: bool = False,
    overwrite_loss_config_cache: bool = False,
):
    # Build each dict and pass to Trainer
    optim_config = {
        "cache": optimizer_config_cache,
        "overwrite_cache": overwrite_optimizer_config_cache,
        "optimizer_type": optimizer_type,
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "dampening": dampening,
        "b1": b1,
        "b2": b2,
        "eps": eps,
        "rho": rho,
    }
    model_config = {
        "cache": model_config_cache,
        "overwrite_cache": overwrite_model_config_cache,
        "model_type": ModelType.e3nn,
        "rand_seed": model_rand_seed,
        "weights_path": weights_path,
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
        "num_layers": num_layers,
        "neighbor_dist": neighbor_dist,
        "num_basis": num_basis,
        "num_radial_layers": num_radial_layers,
        "num_radial_neurons": num_radial_neurons,
        "num_neighbors": num_neighbors,
        "num_nodes": num_nodes,
    }
    es_config = {
        "cache": es_config_cache,
        "overwrite_cache": overwrite_es_config_cache,
        "es_type": es_type,
        "patience": es_patience,
        "n_check": es_n_check,
        "divergence": es_divergence,
    }
    ds_config = {
        "cache": ds_config_cache,
        "overwrite_cache": overwrite_ds_config_cache,
        "exp_file": exp_file,
        "is_structural": True,
        "structures": structures,
        "xtal_regex": xtal_regex,
        "cpd_regex": cpd_regex,
        "cache_file": ds_cache,
        "overwrite": overwrite_ds_cache,
        "grouped": grouped,
        "for_e3nn": True,
    }

    ds_splitter_config = {
        "cache": ds_split_config_cache,
        "overwrite_cache": overwrite_ds_split_config_cache,
        "split_type": ds_split_type,
        "grouped": grouped,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "enforce_one": enforce_one,
        "rand_seed": ds_rand_seed,
    }
    loss_config = {
        "cache": loss_config_cache,
        "overwrite_cache": overwrite_loss_config_cache,
        "loss_type": loss_type,
        "semiquant_fill": semiquant_fill,
    }

    # Parse loss_dict
    if loss_dict:
        loss_dict = json.loads(loss_dict.read_text())

    trainer_kwargs = {
        "optimizer_config": optim_config,
        "model_config": model_config,
        "es_config": es_config,
        "ds_config": ds_config,
        "ds_splitter_config": ds_splitter_config,
        "loss_config": loss_config,
        "auto_init": auto_init,
        "start_epoch": start_epoch,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "target_prop": target_prop,
        "cont": cont,
        "loss_dict": loss_dict,
        "device": device,
        "output_dir": output_dir,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "wandb_name": wandb_name,
        "extra_config": extra_config,
    }

    sweep_kwargs = {
        "sweep_config": sweep_config,
        "force_new_sweep": force_new_sweep,
    }

    sweeper = _build_sweeper(
        trainer_kwargs,
        sweep_kwargs,
        trainer_config_cache,
        sweep_config_cache,
        overwrite_sweep_config_cache,
    )

    sweeper.start_continue_sweep()


@sweep.command(name="visnet")
@output_dir
@weights_path
@trainer_config_cache
@sweep_config_cache
@optim_args
@model_config_cache
@model_rand_seed
@wandb_args
@mtenn_args
@visnet_args
@es_args
@graph_ds_args
@struct_ds_args
@ds_split_args
@loss_args
@trainer_args
@sweep_config
@force_new_sweep
@sweep_config_cache_overwrite
@overwrite_args
def sweep_visnet(
    output_dir: Path | None = None,
    weights_path: Path | None = None,
    trainer_config_cache: Path | None = None,
    sweep_config_cache: Path | None = None,
    optimizer_type: OptimizerType | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    momentum: float | None = None,
    dampening: float | None = None,
    b1: float | None = None,
    b2: float | None = None,
    eps: float | None = None,
    rho: float | None = None,
    optimizer_config_cache: Path | None = None,
    use_wandb: bool | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    extra_config: list[str] | None = None,
    grouped: bool | None = None,
    strategy: StrategyConfig | None = None,
    pred_readout: ReadoutConfig | None = None,
    combination: CombinationConfig | None = None,
    comb_readout: ReadoutConfig | None = None,
    max_comb_neg: bool | None = None,
    max_comb_scale: float | None = None,
    pred_substrate: float | None = None,
    pred_km: float | None = None,
    comb_substrate: float | None = None,
    comb_km: float | None = None,
    model_config_cache: Path | None = None,
    model_rand_seed: int | None = None,
    lmax: int | None = None,
    vecnorm_type: str | None = None,
    trainable_vecnorm: bool | None = None,
    num_heads: int | None = None,
    num_layers: int | None = None,
    hidden_channels: int | None = None,
    num_rbf: int | None = None,
    trainable_rbf: bool | None = None,
    max_z: int | None = None,
    cutoff: float | None = None,
    max_num_neighbors: int | None = None,
    vertex: bool | None = None,
    reduce_op: str | None = None,
    mean: float | None = None,
    std: float | None = None,
    derivative: bool | None = None,
    es_type: EarlyStoppingType | None = None,
    es_patience: int | None = None,
    es_n_check: int | None = None,
    es_divergence: float | None = None,
    es_config_cache: Path | None = None,
    exp_file: Path | None = None,
    ds_cache: Path | None = None,
    ds_config_cache: Path | None = None,
    structures: str | None = None,
    xtal_regex: str = MPRO_ID_REGEX,
    cpd_regex: str = MOONSHOT_CDD_ID_REGEX,
    ds_split_type: DatasetSplitterType | None = None,
    train_frac: float | None = None,
    val_frac: float | None = None,
    test_frac: float | None = None,
    enforce_one: bool | None = None,
    ds_rand_seed: int | None = None,
    ds_split_config_cache: Path | None = None,
    loss_type: LossFunctionType | None = None,
    semiquant_fill: float | None = None,
    loss_config_cache: Path | None = None,
    auto_init: bool | None = None,
    start_epoch: int | None = None,
    n_epochs: int | None = None,
    batch_size: int | None = None,
    target_prop: str | None = None,
    cont: bool | None = None,
    loss_dict: dict | None = None,
    device: torch.device | None = None,
    sweep_config: Path | None = None,
    force_new_sweep: bool | None = None,
    overwrite_sweep_config_cache: bool = False,
    overwrite_trainer_config_cache: bool = False,
    overwrite_optimizer_config_cache: bool = False,
    overwrite_model_config_cache: bool = False,
    overwrite_es_config_cache: bool = False,
    overwrite_ds_config_cache: bool = False,
    overwrite_ds_cache: bool = False,
    overwrite_ds_split_config_cache: bool = False,
    overwrite_loss_config_cache: bool = False,
):
    # Build each dict and pass to Trainer
    optim_config = {
        "cache": optimizer_config_cache,
        "overwrite_cache": overwrite_optimizer_config_cache,
        "optimizer_type": optimizer_type,
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "dampening": dampening,
        "b1": b1,
        "b2": b2,
        "eps": eps,
        "rho": rho,
    }
    model_config = {
        "cache": model_config_cache,
        "overwrite_cache": overwrite_model_config_cache,
        "model_type": ModelType.visnet,
        "rand_seed": model_rand_seed,
        "weights_path": weights_path,
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
        "lmax": lmax,
        "vecnorm_type": vecnorm_type,
        "trainable_vecnorm": trainable_vecnorm,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "hidden_channels": hidden_channels,
        "num_rbf": num_rbf,
        "trainable_rbf": trainable_rbf,
        "max_z": max_z,
        "cutoff": cutoff,
        "max_num_neighbors": max_num_neighbors,
        "vertex": vertex,
        "reduce_op": reduce_op,
        "mean": mean,
        "std": std,
        "derivative": derivative,
    }
    es_config = {
        "cache": es_config_cache,
        "overwrite_cache": overwrite_es_config_cache,
        "es_type": es_type,
        "patience": es_patience,
        "n_check": es_n_check,
        "divergence": es_divergence,
    }
    ds_config = {
        "cache": ds_config_cache,
        "overwrite_cache": overwrite_ds_config_cache,
        "exp_file": exp_file,
        "is_structural": True,
        "structures": structures,
        "xtal_regex": xtal_regex,
        "cpd_regex": cpd_regex,
        "cache_file": ds_cache,
        "overwrite": overwrite_ds_cache,
        "grouped": grouped,
        "for_e3nn": False,
    }

    ds_splitter_config = {
        "cache": ds_split_config_cache,
        "overwrite_cache": overwrite_ds_split_config_cache,
        "split_type": ds_split_type,
        "grouped": grouped,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "enforce_one": enforce_one,
        "rand_seed": ds_rand_seed,
    }
    loss_config = {
        "cache": loss_config_cache,
        "overwrite_cache": overwrite_loss_config_cache,
        "loss_type": loss_type,
        "semiquant_fill": semiquant_fill,
    }

    # Parse loss_dict
    if loss_dict:
        loss_dict = json.loads(loss_dict.read_text())

    trainer_kwargs = {
        "optimizer_config": optim_config,
        "model_config": model_config,
        "es_config": es_config,
        "ds_config": ds_config,
        "ds_splitter_config": ds_splitter_config,
        "loss_config": loss_config,
        "auto_init": auto_init,
        "start_epoch": start_epoch,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "target_prop": target_prop,
        "cont": cont,
        "loss_dict": loss_dict,
        "device": device,
        "output_dir": output_dir,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "wandb_name": wandb_name,
        "extra_config": extra_config,
    }

    sweep_kwargs = {
        "sweep_config": sweep_config,
        "force_new_sweep": force_new_sweep,
    }

    sweeper = _build_sweeper(
        trainer_kwargs,
        sweep_kwargs,
        trainer_config_cache,
        sweep_config_cache,
        overwrite_sweep_config_cache,
    )

    sweeper.start_continue_sweep()


def _build_sweeper(
    trainer_kwargs: dict,
    sweep_kwargs: dict,
    trainer_config_cache: Path | None = None,
    sweep_config_cache: Path | None = None,
    overwrite_sweep_config_cache: bool = False,
):
    """
    Helper function to build a Sweeper from kwargs and (optionally) a JSON Sweeper
    and/or Trainer config file. If a config file(s) is given, those args will be used as
    the default, to be overwritten by anything in trainer_kwargs or sweep_kwargs.

    Parameters
    ----------
    trainer_kwargs : dict
        Args to be passed to the Trainer constructor. These will supersede anything in
        trainer_config_cache
    sweep_kwargs : dict
        Args to be passed to the Sweeper constructor. These will supersede anything in
        sweep_config_cache
    trainer_config_cache : Path, optional
        Trainer Config JSON cache file. Any other CLI args that are passed will
        supersede anything in this file
    sweep_config_cache : Path, optional
        Sweeper Config JSON cache file. Any other CLI args that are passed will
        supersede anything in this file
    overwrite_sweep_config_cache : bool, default=False
        Overwrite any existing Sweeper JSON cache file

    Returns
    -------
    Sweeper
    """

    # Filter out None Trainer kwargs
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if v is not None}

    # If we got a config for the Trainer, load those args and merge with CLI args
    if trainer_config_cache and trainer_config_cache.exists():
        print("loading trainer args from cache", flush=True)
        config_trainer_kwargs = json.loads(trainer_config_cache.read_text())

        for config_name, config_val in config_trainer_kwargs.items():
            # Arg wasn't passed at all, so got filtered out before
            if config_name not in trainer_kwargs:
                continue

            if isinstance(config_val, dict):
                config_val.update(
                    {
                        k: v
                        for k, v in trainer_kwargs[config_name].items()
                        if v is not None
                    }
                )
            else:
                config_trainer_kwargs[config_name] = trainer_kwargs[config_name]

        trainer_kwargs = config_trainer_kwargs

    # Filter out None Sweeper kwargs
    sweep_kwargs = {k: v for k, v in sweep_kwargs.items() if v is not None}

    # If we got a config for the Sweeper, load those args and merge with CLI args
    if sweep_config_cache and sweep_config_cache.exists():
        print("loading sweep args from cache", flush=True)
        config_sweep_kwargs = json.loads(sweep_config_cache.read_text())

        for config_name, config_val in config_sweep_kwargs.items():
            # Arg wasn't passed at all, so got filtered out before
            if config_name not in sweep_kwargs:
                continue

            if isinstance(config_val, dict):
                config_val.update(
                    {
                        k: v
                        for k, v in sweep_kwargs[config_name].items()
                        if v is not None
                    }
                )
            else:
                config_sweep_kwargs[config_name] = sweep_kwargs[config_name]

        sweep_kwargs = config_sweep_kwargs

    try:
        sweeper = Sweeper(**sweep_kwargs, **trainer_kwargs)
    except pydantic.ValidationError as exc:
        # Only want to handle missing values, so if anything else went wrong just raise
        #  the pydantic error
        if any([err["type"] != "value_error.missing" for err in exc.errors()]):
            raise exc

        # Gather all missing values
        missing_vals = [err["loc"][0] for err in exc.errors()]

        raise ValueError(
            "Tried to build Sweeper but missing required values: ["
            + ", ".join(missing_vals)
            + "]"
        )

    # Save full config
    if sweep_config_cache and (
        (not sweep_config_cache.exists()) or overwrite_sweep_config_cache
    ):
        sweep_config_cache.write_text(sweeper.json())

    return sweeper
