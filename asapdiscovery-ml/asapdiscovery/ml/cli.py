import json
from glob import glob
from pathlib import Path

import click
import pydantic
import torch
from asapdiscovery.data.schema import ExperimentalCompoundData
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.utils import (
    MOONSHOT_CDD_ID_REGEX,
    MPRO_ID_REGEX,
    extract_compounds_from_filenames,
)
from asapdiscovery.ml.cli_args import (
    ds_split_args,
    e3nn_args,
    es_args,
    gat_args,
    graph_ds_args,
    loss_args,
    model_config_cache,
    mtenn_args,
    optim_args,
    output_dir,
    schnet_args,
    struct_ds_args,
    trainer_args,
    trainer_config_cache,
    wandb_args,
)
from asapdiscovery.ml.schema_v2.config import (
    DatasetConfig,
    DatasetSplitterConfig,
    DatasetSplitterType,
    DatasetType,
    EarlyStoppingConfig,
    EarlyStoppingType,
    LossFunctionConfig,
    LossFunctionType,
    OptimizerConfig,
    OptimizerType,
)
from asapdiscovery.ml.schema_v2.trainer import Trainer
from mtenn.config import (
    CombinationConfig,
    E3NNModelConfig,
    GATModelConfig,
    ReadoutConfig,
    SchNetModelConfig,
    StrategyConfig,
)


@click.group()
def cli():
    pass


# Functions for just building a Trainer and then dumping it
@click.group()
def build():
    pass


# Function for training using an already built Trainer
@cli.command()
def train():
    pass


# Functions for building a Trainer and subsequently training the model
@click.group(name="build-and-train")
def build_and_train():
    pass


cli.add_command(build)
cli.add_command(build_and_train)


@build.command(name="gat")
def build_gat():
    pass


@build.command(name="schnet")
def build_schnet():
    pass


@build.command(name="e3nn")
def build_e3nn():
    pass


@build_and_train.command(name="gat")
@output_dir
@trainer_config_cache
@optim_args
@wandb_args
@model_config_cache
@mtenn_args
@gat_args
@es_args
@graph_ds_args
@ds_split_args
@loss_args
@trainer_args
def build_and_train_gat(
    output_dir: Path,
    trainer_config_cache: Path | None = None,
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
    use_wandb: bool = False,
    sweep: bool = False,
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
    rand_seed: int | None = None,
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
):
    t = build_trainer_gat(
        output_dir,
        trainer_config_cache,
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
        use_wandb,
        sweep,
        wandb_project,
        wandb_name,
        extra_config,
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
        model_config_cache,
        in_feats,
        num_layers,
        hidden_feats,
        num_heads,
        feat_drops,
        attn_drops,
        alphas,
        residuals,
        agg_modes,
        biases,
        allow_zero_in_degree,
        es_type,
        es_patience,
        es_n_check,
        es_divergence,
        es_config_cache,
        exp_file,
        ds_cache,
        ds_config_cache,
        ds_split_type,
        train_frac,
        val_frac,
        test_frac,
        enforce_one,
        rand_seed,
        ds_split_config_cache,
        loss_type,
        semiquant_fill,
        loss_config_cache,
        auto_init,
        start_epoch,
        n_epochs,
        batch_size,
        target_prop,
        cont,
        loss_dict,
        device,
    )

    t.initialize()
    model, loss_dict = t.train()

    print("done", flush=True)


@build_and_train.command(name="schnet")
@output_dir
@trainer_config_cache
@optim_args
@model_config_cache
@wandb_args
@mtenn_args
@schnet_args
@es_args
@graph_ds_args
@struct_ds_args
@ds_split_args
@loss_args
@trainer_args
def build_and_train_schnet(
    output_dir: Path,
    trainer_config_cache: Path | None = None,
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
    use_wandb: bool = False,
    sweep: bool = False,
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
    rand_seed: int | None = None,
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
):
    t = build_trainer_schnet(
        output_dir,
        trainer_config_cache,
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
        use_wandb,
        sweep,
        wandb_project,
        wandb_name,
        extra_config,
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
        model_config_cache,
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
        es_type,
        es_patience,
        es_n_check,
        es_divergence,
        es_config_cache,
        exp_file,
        ds_cache,
        ds_config_cache,
        structures,
        xtal_regex,
        cpd_regex,
        ds_split_type,
        train_frac,
        val_frac,
        test_frac,
        enforce_one,
        rand_seed,
        ds_split_config_cache,
        loss_type,
        semiquant_fill,
        loss_config_cache,
        auto_init,
        start_epoch,
        n_epochs,
        batch_size,
        target_prop,
        cont,
        loss_dict,
        device,
    )

    t.initialize()
    model, loss_dict = t.train()

    print("done", flush=True)


@build_and_train.command("e3nn")
@output_dir
@trainer_config_cache
@optim_args
@model_config_cache
@wandb_args
@mtenn_args
@e3nn_args
@es_args
@graph_ds_args
@struct_ds_args
@ds_split_args
@loss_args
@trainer_args
def build_and_train_e3nn(
    output_dir: Path,
    trainer_config_cache: Path | None = None,
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
    use_wandb: bool = False,
    sweep: bool = False,
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
    rand_seed: int | None = None,
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
):
    t = build_trainer_e3nn(
        output_dir,
        trainer_config_cache,
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
        use_wandb,
        sweep,
        wandb_project,
        wandb_name,
        extra_config,
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
        model_config_cache,
        num_atom_types,
        irreps_hidden,
        lig,
        irreps_edge_attr,
        num_layers,
        neighbor_dist,
        num_basis,
        num_radial_layers,
        num_radial_neurons,
        num_neighbors,
        num_nodes,
        es_type,
        es_patience,
        es_n_check,
        es_divergence,
        es_config_cache,
        exp_file,
        ds_cache,
        ds_config_cache,
        structures,
        xtal_regex,
        cpd_regex,
        ds_split_type,
        train_frac,
        val_frac,
        test_frac,
        enforce_one,
        rand_seed,
        ds_split_config_cache,
        loss_type,
        semiquant_fill,
        loss_config_cache,
        auto_init,
        start_epoch,
        n_epochs,
        batch_size,
        target_prop,
        cont,
        loss_dict,
        device,
    )

    t.initialize()
    model, loss_dict = t.train()

    print("done", flush=True)


def build_trainer_gat(
    output_dir: Path,
    trainer_config_cache: Path | None = None,
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
    use_wandb: bool = False,
    sweep: bool = False,
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
    rand_seed: int | None = None,
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
):
    # First check if Trainer cache exists and skip everything else if so
    if trainer_config_cache and trainer_config_cache.exists():
        trainer_kwargs = json.loads(trainer_config_cache.read_text())
        trainer_kwargs["device"] = torch.device(trainer_kwargs["device"])
        trainer_kwargs["model_config"] = GATModelConfig(
            **trainer_kwargs["model_config"]
        )
        print("loaded trainer from cache", flush=True)
        t = Trainer(**trainer_kwargs)
    else:
        optim_config = _build_arbitrary_config(
            config_cls=OptimizerConfig,
            config_file=optimizer_config_cache,
            optimizer_type=optimizer_type,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            dampening=dampening,
            b1=b1,
            b2=b2,
            eps=eps,
            rho=rho,
        )
        model_config = _build_arbitrary_config(
            config_cls=GATModelConfig,
            config_file=model_config_cache,
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
        if (es_config_cache and es_config_cache.exists()) or es_type:
            es_config = _build_arbitrary_config(
                config_cls=EarlyStoppingConfig,
                config_file=es_config_cache,
                es_type=es_type,
                es_patience=es_patience,
                es_n_check=es_n_check,
                es_divergence=es_divergence,
            )
        else:
            es_config = None
        ds_config = _build_ds_config(
            exp_file=exp_file,
            structures=None,
            xtal_regex=None,
            cpd_regex=None,
            ds_cache=ds_cache,
            ds_config_cache=ds_config_cache,
            is_structural=False,
            is_grouped=grouped,
        )
        ds_splitter_config = _build_arbitrary_config(
            config_cls=DatasetSplitterConfig,
            config_file=ds_split_config_cache,
            split_type=ds_split_type,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            enforce_one=enforce_one,
            rand_seed=rand_seed,
        )
        loss_config = _build_arbitrary_config(
            config_cls=LossFunctionConfig,
            config_file=loss_config_cache,
            loss_type=loss_type,
            semiquant_fill=semiquant_fill,
        )

        # Parse loss_dict
        if loss_dict:
            loss_dict = json.loads(loss_dict.read_text())

        # Filter out None Trainer kwargs
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
        }
        trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if v is not None}

        t = Trainer(**trainer_kwargs)

        # Save Trainer
        if trainer_config_cache:
            trainer_config_cache.write_text(t.json())

    return t


def build_trainer_schnet(
    output_dir: Path,
    trainer_config_cache: Path | None = None,
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
    use_wandb: bool = False,
    sweep: bool = False,
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
    rand_seed: int | None = None,
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
):
    # First check if Trainer cache exists and skip everything else if so
    if trainer_config_cache and trainer_config_cache.exists():
        trainer_kwargs = json.loads(trainer_config_cache.read_text())
        trainer_kwargs["device"] = torch.device(trainer_kwargs["device"])
        trainer_kwargs["model_config"] = SchNetModelConfig(
            **trainer_kwargs["model_config"]
        )
        print("loaded trainer from cache", flush=True)
        t = Trainer(**trainer_kwargs)
    else:
        optim_config = _build_arbitrary_config(
            config_cls=OptimizerConfig,
            config_file=optimizer_config_cache,
            optimizer_type=optimizer_type,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            dampening=dampening,
            b1=b1,
            b2=b2,
            eps=eps,
            rho=rho,
        )
        model_config = _build_arbitrary_config(
            config_cls=SchNetModelConfig,
            config_file=model_config_cache,
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
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            readout=readout,
            dipole=dipole,
            mean=mean,
            std=std,
        )
        if (es_config_cache and es_config_cache.exists()) or es_type:
            es_config = _build_arbitrary_config(
                config_cls=EarlyStoppingConfig,
                config_file=es_config_cache,
                es_type=es_type,
                es_patience=es_patience,
                es_n_check=es_n_check,
                es_divergence=es_divergence,
            )
        else:
            es_config = None
        ds_config = _build_ds_config(
            exp_file=exp_file,
            structures=structures,
            xtal_regex=xtal_regex,
            cpd_regex=cpd_regex,
            ds_cache=ds_cache,
            ds_config_cache=ds_config_cache,
            is_structural=True,
            is_grouped=grouped,
        )
        ds_splitter_config = _build_arbitrary_config(
            config_cls=DatasetSplitterConfig,
            config_file=ds_split_config_cache,
            split_type=ds_split_type,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            enforce_one=enforce_one,
            rand_seed=rand_seed,
        )
        loss_config = _build_arbitrary_config(
            config_cls=LossFunctionConfig,
            config_file=loss_config_cache,
            loss_type=loss_type,
            semiquant_fill=semiquant_fill,
        )

        # Parse loss_dict
        if loss_dict:
            loss_dict = json.loads(loss_dict.read_text())

        # Filter out None Trainer kwargs
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
        }
        trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if v is not None}

        t = Trainer(**trainer_kwargs)

        # Save Trainer
        if trainer_config_cache:
            trainer_config_cache.write_text(t.json())

    return t


def build_trainer_e3nn(
    output_dir: Path,
    trainer_config_cache: Path | None = None,
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
    use_wandb: bool = False,
    sweep: bool = False,
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
    rand_seed: int | None = None,
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
):
    # First check if Trainer cache exists and skip everything else if so
    if trainer_config_cache and trainer_config_cache.exists():
        trainer_kwargs = json.loads(trainer_config_cache.read_text())
        trainer_kwargs["device"] = torch.device(trainer_kwargs["device"])
        trainer_kwargs["model_config"] = E3NNModelConfig(
            **trainer_kwargs["model_config"]
        )
        print("loaded trainer from cache", flush=True)
        t = Trainer(**trainer_kwargs)
    else:
        optim_config = _build_arbitrary_config(
            config_cls=OptimizerConfig,
            config_file=optimizer_config_cache,
            optimizer_type=optimizer_type,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            dampening=dampening,
            b1=b1,
            b2=b2,
            eps=eps,
            rho=rho,
        )
        model_config = _build_arbitrary_config(
            config_cls=E3NNModelConfig,
            config_file=model_config_cache,
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
            num_atom_types=num_atom_types,
            irreps_hidden=irreps_hidden,
            lig=lig,
            irreps_edge_attr=irreps_edge_attr,
            num_layers=num_layers,
            neighbor_dist=neighbor_dist,
            num_basis=num_basis,
            num_radial_layers=num_radial_layers,
            num_radial_neurons=num_radial_neurons,
            num_neighbors=num_neighbors,
            num_nodes=num_nodes,
        )
        if (es_config_cache and es_config_cache.exists()) or es_type:
            es_config = _build_arbitrary_config(
                config_cls=EarlyStoppingConfig,
                config_file=es_config_cache,
                es_type=es_type,
                es_patience=es_patience,
                es_n_check=es_n_check,
                es_divergence=es_divergence,
            )
        else:
            es_config = None
        ds_config = _build_ds_config(
            exp_file=exp_file,
            structures=structures,
            xtal_regex=xtal_regex,
            cpd_regex=cpd_regex,
            ds_cache=ds_cache,
            ds_config_cache=ds_config_cache,
            is_structural=True,
            is_grouped=grouped,
        )
        ds_splitter_config = _build_arbitrary_config(
            config_cls=DatasetSplitterConfig,
            config_file=ds_split_config_cache,
            split_type=ds_split_type,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            enforce_one=enforce_one,
            rand_seed=rand_seed,
        )
        loss_config = _build_arbitrary_config(
            config_cls=LossFunctionConfig,
            config_file=loss_config_cache,
            loss_type=loss_type,
            semiquant_fill=semiquant_fill,
        )

        # Parse loss_dict
        if loss_dict:
            loss_dict = json.loads(loss_dict.read_text())

        # Filter out None Trainer kwargs
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
        }
        trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if v is not None}

        t = Trainer(**trainer_kwargs)

        # Save Trainer
        if trainer_config_cache:
            trainer_config_cache.write_text(t.json())

    return t


def _check_ds_args(exp_file, structures, ds_cache, ds_config_cache, is_structural):
    """
    Helper function to check that all necessary dataset files were passed.

    Parameters
    ----------
    exp_file : Path
        JSON file giving a list of ExperimentalDataCompound objects
    structures : Path
        Glob or directory containing PDB files
    ds_cache : Path
        Dataset cache file
    ds_config_cache : Path
        Dataset config cache function

    Returns
    -------
    bool
        Whether an appropriate combination of args was passed
    """
    # Can just load from the config cache file so don't need anything else
    if ds_config_cache and ds_config_cache.exists():
        return True

    # Otherwise need to load data so make sure they all exist
    if (not exp_file) or (not exp_file.exists()):
        return False
    if is_structural:
        if not structures:
            return False
        if Path(structures).is_dir():
            # Make sure there's at least one PDB file
            try:
                _ = next(iter(Path(structures).glob("*.pdb")))
            except StopIteration:
                return False
        else:
            # Make sure there's at least one file that matches the glob
            try:
                _ = next(iter(glob(structures)))
            except StopIteration:
                return False

    # Nothing has failed so we should be good to go
    return True


def _build_ds_config(
    exp_file,
    structures,
    xtal_regex,
    cpd_regex,
    ds_cache,
    ds_config_cache,
    is_structural,
    is_grouped,
):
    """
    Helper function to build a DatasetConfig object.

    Parameters
    ----------
    exp_file : Path
        JSON file giving a list of ExperimentalDataCompound objects
    structures : Path
        Glob or directory containing PDB files
    ds_cache : Path
        Dataset cache file
    ds_config_cache : Path
        Dataset config cache function

    Returns
    -------
    DatasetConfig
        DatasetConfig object
    """

    if not _check_ds_args(
        exp_file=exp_file,
        structures=None,
        ds_cache=ds_cache,
        ds_config_cache=ds_config_cache,
        is_structural=False,
    ):
        raise ValueError("Invalid combination of dataset args.")

    if ds_config_cache and ds_config_cache.exists():
        print("loading from cache", flush=True)
        return DatasetConfig(**json.loads(ds_config_cache.read_text()))

    # Pick correct DatasetType
    if is_structural:
        ds_type = DatasetType.structural
    else:
        ds_type = DatasetType.graph

    # Parse experimental data
    exp_compounds = [
        ExperimentalCompoundData(**d) for d in json.loads(exp_file.read_text())
    ]
    exp_data = {
        c.compound_id: c.experimental_data | {"date_created": c.date_created}
        for c in exp_compounds
    }

    # Create Ligand/Complex objects
    if is_structural:
        if Path(structures).is_dir():
            all_str_fns = Path(structures).glob("*.pdb")
        else:
            all_str_fns = glob(structures)
        compounds = extract_compounds_from_filenames(
            all_str_fns, xtal_pat=xtal_regex, compound_pat=cpd_regex, fail_val="NA"
        )
        print(len(list(all_str_fns)), len(compounds), flush=True)
        input_data = [
            Complex.from_pdb(
                fn,
                target_kwargs={"target_name": cpd[0]},
                ligand_kwargs={"compound_name": cpd[1]},
            )
            for fn, cpd in zip(all_str_fns, compounds)
        ]
    else:
        input_data = [
            Ligand.from_smiles(c.smiles, compound_name=c.compound_id)
            for c in exp_compounds
        ]

    config_kwargs = {
        "ds_type": ds_type,
        "exp_data": exp_data,
        "input_data": input_data,
        "cache_file": ds_cache,
    }
    if is_grouped is not None:
        config_kwargs["grouped"] = is_grouped
    ds_config = DatasetConfig(**config_kwargs)

    # Save file if desired
    if ds_config_cache:
        ds_config_cache.write_text(ds_config.json())

    return ds_config


def _build_arbitrary_config(config_cls, config_file, **config_kwargs):
    """
    Helper function to load/build an arbitrary Config object. All kwargs in
    config_kwargs will overwrite anything in config_file, and everything will be passed
    to the config_cls constructor, so make sure only the appropriate kwargs are passed.

    Parameters
    ----------
    config_cls : type
        Config class. Can in theory be any pydantic schema
    config_file : Path
        Path to config file. Will be loaded if it exists, otherwise will be saved after
        object creation.
    config_kwargs : dict
        Dict giving all CLI args for Config construction. Will discard any that are None
        to allow the Config defaults to kick in.

    Returns
    -------
    config_cls
        Instance of whatever class is passed
    """

    if config_file and config_file.exists():
        print("loading from cache", config_cls, flush=True)
        loaded_kwargs = json.loads(config_file.read_text())
    else:
        loaded_kwargs = {}

    # Filter out None kwargs so defaults kick in
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

    # Update stored config args
    loaded_kwargs |= config_kwargs

    # Build Config, catching and handling missing required values
    try:
        config = config_cls(**loaded_kwargs)
    except pydantic.ValidationError as exc:
        # Only want to handle missing values, so if anything else went wrong just raise
        #  the pydantic error
        if any([err["type"] != "value_error.missing" for err in exc.errors()]):
            raise exc

        # Gather all missing values
        missing_vals = [err["loc"][0] for err in exc.errors()]

        raise ValueError(
            f"Tried to build {config_cls} but missing required values: ["
            + ", ".join(missing_vals)
            + "]"
        )

    # If a non-existent file was passed, store the Config
    if config_file:
        config_file.write_text(config.json())

    return config