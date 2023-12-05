import json
from glob import glob
from pathlib import Path

import click
from asapdiscovery.data.schema import ExperimentalCompoundData
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.utils import (
    MOONSHOT_CDD_ID_REGEX,
    MPRO_ID_REGEX,
    extract_compounds_from_filenames,
)
from asapdiscovery.ml.cli_args import (
    config_file,
    ds_cache,
    ds_config_cache,
    ds_split_args,
    e3nn_args,
    es_args,
    exp_file,
    gat_args,
    mtenn_args,
    optim_args,
    output_dir,
    schnet_args,
    str_files,
    str_fn_cpd_regex,
    str_fn_xtal_regex,
    wandb_args,
)
from asapdiscovery.ml.schema_v2.config import (
    DatasetConfig,
    DatasetSplitterConfig,
    DatasetSplitterType,
    DatasetType,
    EarlyStoppingConfig,
    EarlyStoppingType,
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
@exp_file
@ds_cache
@ds_config_cache
@optim_args
@config_file
@wandb_args
@mtenn_args
@gat_args
@es_args
@ds_split_args
def build_and_train_gat(
    output_dir: Path,
    exp_file: Path | None = None,
    ds_cache: Path | None = None,
    ds_config_cache: Path | None = None,
    optimizer_type: OptimizerType | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    momentum: float | None = None,
    dampening: float | None = None,
    b1: float | None = None,
    b2: float | None = None,
    eps: float | None = None,
    rho: float | None = None,
    config_file: Path | None = None,
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
    split_type: DatasetSplitterType | None = None,
    train_frac: float | None = None,
    val_frac: float | None = None,
    test_frac: float | None = None,
    enforce_1: bool | None = None,
    rand_seed: int | None = None,
    ds_split_config_cache: Path | None = None,
):
    optim_config = OptimizerConfig(
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
    model_config = _build_model_config(
        GATModelConfig,
        config_file=config_file,
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
    es_config = _build_es_config(
        es_config_cache=es_config_cache,
        es_type=es_type,
        es_patience=es_patience,
        es_n_check=es_n_check,
        es_divergence=es_divergence,
    )
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

    return Trainer(
        optimizer_config=optim_config,
        model_config=model_config,
        es_config=es_config,
        ds_config=ds_config,
    )


@build_and_train.command(name="schnet")
@output_dir
@exp_file
@str_files
@str_fn_cpd_regex
@str_fn_xtal_regex
@ds_cache
@ds_config_cache
@optim_args
@config_file
@wandb_args
@mtenn_args
@schnet_args
@es_args
@ds_split_args
def build_and_train_schnet(
    output_dir: Path,
    exp_file: Path | None = None,
    structures: str | None = None,
    cpd_regex: str = MOONSHOT_CDD_ID_REGEX,
    xtal_regex: str = MPRO_ID_REGEX,
    ds_cache: Path | None = None,
    ds_config_cache: Path | None = None,
    optimizer_type: OptimizerType | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    momentum: float | None = None,
    dampening: float | None = None,
    b1: float | None = None,
    b2: float | None = None,
    eps: float | None = None,
    rho: float | None = None,
    config_file: Path | None = None,
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
    split_type: DatasetSplitterType | None = None,
    train_frac: float | None = None,
    val_frac: float | None = None,
    test_frac: float | None = None,
    enforce_1: bool | None = None,
    rand_seed: int | None = None,
    ds_split_config_cache: Path | None = None,
):
    optim_config = OptimizerConfig(
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
    model_config = _build_model_config(
        SchNetModelConfig,
        config_file=config_file,
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
    es_config = _build_es_config(
        es_config_cache=es_config_cache,
        es_type=es_type,
        es_patience=es_patience,
        es_n_check=es_n_check,
        es_divergence=es_divergence,
    )
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

    return Trainer(
        optimizer_config=optim_config,
        model_config=model_config,
        es_config=es_config,
        ds_config=ds_config,
    )


@build_and_train.command("e3nn")
@output_dir
@exp_file
@str_files
@str_fn_cpd_regex
@str_fn_xtal_regex
@ds_cache
@ds_config_cache
@optim_args
@config_file
@wandb_args
@mtenn_args
@e3nn_args
@es_args
@ds_split_args
def build_and_train_e3nn(
    output_dir: Path,
    exp_file: Path | None = None,
    structures: str | None = None,
    cpd_regex: str = MOONSHOT_CDD_ID_REGEX,
    xtal_regex: str = MPRO_ID_REGEX,
    ds_cache: Path | None = None,
    ds_config_cache: Path | None = None,
    optimizer_type: OptimizerType | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    momentum: float | None = None,
    dampening: float | None = None,
    b1: float | None = None,
    b2: float | None = None,
    eps: float | None = None,
    rho: float | None = None,
    config_file: Path | None = None,
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
    split_type: DatasetSplitterType | None = None,
    train_frac: float | None = None,
    val_frac: float | None = None,
    test_frac: float | None = None,
    enforce_1: bool | None = None,
    rand_seed: int | None = None,
    ds_split_config_cache: Path | None = None,
):
    optim_config = OptimizerConfig(
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
    model_config = _build_model_config(
        E3NNModelConfig,
        config_file=config_file,
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
    es_config = _build_es_config(
        es_config_cache=es_config_cache,
        es_type=es_type,
        es_patience=es_patience,
        es_n_check=es_n_check,
        es_divergence=es_divergence,
    )
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

    return Trainer(
        optimizer_config=optim_config,
        model_config=model_config,
        es_config=es_config,
        ds_config=ds_config,
    )


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


def _build_model_config(config_cls, config_file, **model_kwargs):
    """
    Helper function to load/build a model config object.

    Parameters
    ----------
    config_cls : type
        ModelConfigBase subclass
    config_file : Path
        Path to config file. Will be loaded if it exists, otherwise will be saved after
        loading.
    model_kwargs : dict
        Dict giving all CLI args for model construction. Will discard any that are None
        to allow the Config defaults to kick in.

    Returns
    -------
    config_cls
        Instance of whatever class is passed
    """

    # Get file config options
    if config_file and config_file.exists():
        model_config = json.loads(config_file.read_text())
    else:
        model_config = {}

    # Filter out passed kwargs that are None
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    # Update model_config
    model_config |= model_kwargs

    # Construct Config object
    model_config = config_cls(model_config)

    # Save if desired
    if config_file and (not config_file.exists()):
        config_file.write_text(model_config.json())

    return model_config


def _build_es_config(es_config_cache, **es_kwargs):
    """
    Helper function to load/build an EarlyStoppingConfig object.

    Parameters
    ----------
    es_config_cache : Path | None
        Path giving a JSON file containing a serialized EarlyStoppingConfig object. Any
        other kwargs passed in es_kwargs will supersede anything in this Config.
    es_kwargs : dict
        Dict giving all CLI args for model construction. Will discard any that are None
        to allow the Config defaults to kick in.

    Returns
    -------
    config_cls
        Instance of whatever class is passed
    """
    if es_config_cache and es_config_cache.exists():
        print("loading from cache", flush=True)
        loaded_kwargs = json.loads(es_config_cache.read_text())
    else:
        loaded_kwargs = {}

    # Filter out None kwargs so defaults kick in
    es_kwargs = {k: v for k, v in es_kwargs if v is not None}

    # Update stored config args
    loaded_kwargs |= es_kwargs

    # Build Config
    es_config = EarlyStoppingConfig(**loaded_kwargs)

    # If a non-existent file was passed, store the Config
    if es_config_cache:
        es_config_cache.write_text(es_config.json())

    return es_config
