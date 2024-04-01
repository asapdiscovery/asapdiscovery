import json
import pickle as pkl
from glob import glob
from pathlib import Path
from time import time

import numpy as np
import torch
import wandb
from asapdiscovery.data.util.logging import FileLogger
from asapdiscovery.ml.config import (
    DatasetConfig,
    DatasetSplitterConfig,
    EarlyStoppingConfig,
    LossFunctionConfig,
    OptimizerConfig,
)
from mtenn.config import (
    E3NNModelConfig,
    GATModelConfig,
    ModelConfigBase,
    ModelType,
    SchNetModelConfig,
    ViSNetModelConfig,
)
from pydantic import BaseModel, Extra, Field, ValidationError, validator


class Trainer(BaseModel):
    """
    Schema for training an ML model.
    """

    # Required parameters for building the training environment
    optimizer_config: OptimizerConfig = Field(
        ..., description="Config describing the optimizer to use in training."
    )
    model_config: ModelConfigBase = Field(
        ..., description="Config describing the model to train."
    )
    es_config: EarlyStoppingConfig | None = Field(
        None, description="Config describing the early stopping check to use."
    )
    ds_config: DatasetConfig = Field(
        ..., description="Config describing the dataset object to train on."
    )
    ds_splitter_config: DatasetSplitterConfig = Field(
        ...,
        description=(
            "Config describing how to split the dataset into train, val, and "
            "test splits."
        ),
    )
    loss_config: LossFunctionConfig = Field(
        ...,
        description="Config describing the loss function for training.",
    )

    # Options for the training process
    auto_init: bool = Field(
        True,
        description=(
            "Automatically initialize the Trainer if it hasn't already been "
            "done when train is called."
        ),
    )
    start_epoch: int = Field(
        0,
        description="Which epoch to start training at (used for continuing training runs).",
    )
    n_epochs: int = Field(
        300,
        description=(
            "Which epoch to stop training at. For non-continuation runs, this "
            "will be the total number of epochs to train for."
        ),
    )
    batch_size: int = Field(
        1, description="Number of samples to predict on before performing backprop."
    )
    target_prop: str = Field("pIC50", description="Target property to train against.")
    cont: bool = Field(
        False, description="This is a continuation of a previous training run."
    )
    loss_dict: dict = Field({}, description="Dict keeping track of training loss.")
    device: torch.device = Field("cpu", description="Device to train on.")

    # I/O options
    output_dir: Path = Field(
        ...,
        description=(
            "Top-level output directory. A subdirectory with the current W&B "
            "run ID will be made/searched if W&B is being used."
        ),
    )
    log_file: Path | None = Field(
        None,
        description="Output using asapdiscovery.data.FileLogger in addition to stdout.",
    )

    # W&B parameters
    use_wandb: bool = Field(False, description="Use W&B to log model training.")
    sweep: bool = Field(False, description="This run is part of a W&B sweep.")
    wandb_project: str | None = Field(None, description="W&B project name.")
    wandb_name: str | None = Field(None, description="W&B project name.")
    extra_config: dict | None = Field(
        None, description="Any extra config options to log to W&B."
    )

    # Tracker to make sure the optimizer and ML model are built before trying to train
    _is_initialized = False

    class Config:
        # Temporary fix for now. This is necessary for the asapdiscovery Dataset
        #  classes, but we should probably figure out a workaround eventurally. Probably
        #  best to implement __get_validators__ for the Dataset classes.
        arbitrary_types_allowed = True

        # Exclude everything that was built (should be able to fully reconstruct from
        #  the configs)
        fields = {
            "model": {"exclude": True},
            "optimizer": {"exclude": True},
            "es": {"exclude": True},
            "ds": {"exclude": True},
            "ds_train": {"exclude": True},
            "ds_val": {"exclude": True},
            "ds_test": {"exclude": True},
            "loss_func": {"exclude": True},
        }

        # Allow things to be added to the object after initialization/validation
        extra = Extra.allow

        # Custom encoder to cast device to str before trying to serialize
        json_encoders = {torch.device: lambda d: str(d)}

    # Validator to make sure that if output_dir exists, it is a directory
    @validator("output_dir")
    def output_dir_check(cls, p):
        if p.exists():
            assert (
                p.is_dir()
            ), "If given output_dir already exists, it must be a directory."

        return p

    @validator(
        "optimizer_config",
        "model_config",
        "es_config",
        "ds_splitter_config",
        "loss_config",
        pre=True,
    )
    def load_cache_files(cls, config_kwargs, field):
        """
        This validator will load an existing cache file, and update the config with any
        explicitly passed kwargs. If passed, the cache file must be an entry in config
        with the name "cache". This function will also check for the entry
        "overwrite_cache" in config, which, if given and True, will overwrite the given
        cache file.
        """
        config_cls = field.type_

        # If an instance of the actual config class is passed, there's no cache file so
        #  just return
        if isinstance(config_kwargs, config_cls):
            return config_kwargs

        # Some configs are optional so allow Nones (will get caught later if a None
        #  that's not allowed is passed)
        if config_kwargs is None:
            return config_kwargs

        # Special case to handle model_config since the Field annotation is an abstract
        #  class
        if config_cls is ModelConfigBase:
            match config_kwargs["model_type"]:
                case ModelType.GAT:
                    config_cls = GATModelConfig
                case ModelType.schnet:
                    config_cls = SchNetModelConfig
                case ModelType.e3nn:
                    config_cls = E3NNModelConfig
                case ModelType.visnet:
                    config_cls = ViSNetModelConfig
                case other:
                    raise ValueError(
                        f"Can't instantiate model config for type {other}."
                    )

        # Get config cache file and overwrite option (if given). Defaults to no cache
        #  file and not overwriting
        config_file = config_kwargs.pop("cache", None)
        overwrite = config_kwargs.pop("overwrite_cache", False)

        return Trainer._build_arbitrary_config(
            config_cls=config_cls,
            config_file=config_file,
            overwrite=overwrite,
            **config_kwargs,
        )

    @validator("ds_config", pre=True)
    def check_and_build_ds(cls, config_kwargs):
        """
        This validator will first check that the appropriate files exist, and then parse
        the files to construct a DatasetConfig. If passed, the cache file must be an
        entry in config with the name "cache". This function will also check for the
        entry "overwrite_cache" in config, which, if given and True, will overwrite the
        given cache file.
        """

        # If an instance of the actual config class is passed, there's no cache file so
        #  just return
        if isinstance(config_kwargs, DatasetConfig):
            return config_kwargs

        # If a dict version of an existing DatasetConfig is passed, just cast it
        try:
            return DatasetConfig(**config_kwargs)
        except Exception:
            pass

        # Get all the relevant kwarg entries out of config
        ds_config_cache = config_kwargs.pop("cache", None)
        if ds_config_cache:
            ds_config_cache = Path(ds_config_cache)
        overwrite = config_kwargs.pop("overwrite_cache", False)
        exp_file = config_kwargs.pop("exp_file", None)
        if exp_file:
            exp_file = Path(exp_file)
        structures = config_kwargs.pop("structures", "")
        is_structural = config_kwargs.pop("is_structural", False)
        # Pop these here to get them out of config, but only check to make sure they're
        #  not None if we have a structure-based ds
        xtal_regex = config_kwargs.pop("xtal_regex", None)
        cpd_regex = config_kwargs.pop("cpd_regex", None)

        if ds_config_cache and ds_config_cache.exists() and (not overwrite):
            print("loading from cache", flush=True)
            return DatasetConfig(**json.loads(ds_config_cache.read_text()))

        # Can't just load from cache so make sure all the right files exist
        if (not exp_file) or (not exp_file.exists()):
            raise ValueError("Must pass experimental data file.")
        if is_structural:
            if not structures:
                raise ValueError(
                    "Must pass structure files for structure-based dataset."
                )
            if Path(structures).is_dir():
                # Make sure there's at least one PDB file
                try:
                    _ = next(iter(Path(structures).glob("*.pdb")))
                except StopIteration:
                    raise ValueError("No structure files found.")
            else:
                # Make sure there's at least one file that matches the glob
                try:
                    _ = next(iter(glob(structures)))
                except StopIteration:
                    raise ValueError("No structure files found.")

        # Filter out None kwargs so defaults kick in
        config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

        # Pick correct DatasetType
        if is_structural:
            if (xtal_regex is None) or (cpd_regex is None):
                raise ValueError(
                    "Must pass values for xtal_regex and cpd_regex if building a "
                    "structure-based dataset."
                )
            ds_config = DatasetConfig.from_str_files(
                structures=structures,
                xtal_regex=xtal_regex,
                cpd_regex=cpd_regex,
                for_training=True,
                exp_file=exp_file,
                **config_kwargs,
            )
        else:
            ds_config = DatasetConfig.from_exp_file(exp_file, **config_kwargs)

        if ds_config_cache:
            ds_config_cache.write_text(ds_config.json())

        return ds_config

    @staticmethod
    def _build_arbitrary_config(
        config_cls, config_file, overwrite=False, **config_kwargs
    ):
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
        overwrite : bool, default=False
            Don't load from config_file if it exists, and save over it
        config_kwargs : dict
            Dict giving all CLI args for Config construction. Will discard any that are None
            to allow the Config defaults to kick in.

        Returns
        -------
        config_cls
            Instance of whatever class is passed
        """

        if config_file and config_file.exists() and (not overwrite):
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
        except ValidationError as exc:
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
        if config_file and ((not config_file.exists()) or overwrite):
            config_file.write_text(config.json())

        return config

    @validator("device", pre=True)
    def fix_device(cls, v):
        """
        The torch device gets serialized as a string and the Trainer class doesn't
        automatically cast it back to a device.
        """
        return torch.device(v)

    @validator("extra_config", pre=True)
    def parse_extra_config(cls, v):
        """
        This is for compatibility with the CLI, in which these config args will be
        passed as key:value strings. Here we split them up and parse into a dict.
        """

        # Just use the dict that's passed if we're not coming from the CLI
        if isinstance(v, dict):
            return v

        extra_config = {}
        for kvp in v:
            try:
                key, val = kvp.split(":")
            except Exception:
                raise ValueError(f"Couldn't parse key:value pair '{kvp}'.")

            extra_config[key] = val

        return extra_config

    def wandb_init(self):
        """
        Initialize WandB, handling saving the run ID (for continuing the run later).

        Returns
        -------
        str
            The WandB run ID for the initialized run
        """

        if self.sweep:
            run_id = wandb.init().id
        else:
            run_id_fn = self.output_dir / "run_id"

            # Don't serialize input_data for confidentiality/size reasons
            ds_config = self.ds_config.dict()
            del ds_config["input_data"]
            config = self.dict()
            config["ds_config"] = ds_config

            if self.cont:
                # Load run_id to continue from file
                # First make sure the file exists
                if run_id_fn.exists():
                    run_id = run_id_fn.read_text().strip()
                else:
                    raise FileNotFoundError(
                        "Couldn't find run_id file to continue run."
                    )
                # Make sure the run_id is legit
                try:
                    wandb.init(project=self.wandb_project, id=run_id, resume="must")
                except wandb.errors.UsageError:
                    raise wandb.errors.UsageError(
                        f"Run in run_id file ({run_id}) doesn't exist"
                    )
                # Update run config to reflect it's been resumed
                wandb.config.update(config, allow_val_change=True)
            else:
                # Start new run
                run_id = wandb.init(
                    project=self.wandb_project,
                    config=config,
                    name=self.wandb_name,
                ).id

                # Save run_id in case we want to continue later
                if not self.output_dir.exists():
                    print(
                        "No output directory specified, not saving run_id anywhere.",
                        flush=True,
                    )
                else:
                    run_id_fn.write_text(run_id)

                for split, table in zip(
                    ["train", "val", "test"], self._make_wandb_ds_tables()
                ):
                    wandb.log({f"dataset_splits/{split}": table})

        return run_id

    def initialize(self):
        """
        Build the Optimizer and ML Model described by the stored config.
        """

        # Set up FileLogger
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self.logger = FileLogger(
                logname="Trainer",
                path=str(self.log_file.parent),
                logfile=str(self.log_file.name),
            ).getLogger()

        # Build dataset and split
        self.ds = self.ds_config.build()
        self.ds_train, self.ds_val, self.ds_test = self.ds_splitter_config.split(
            self.ds
        )

        # Adjust output_dir and make sure it exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Start the W&B process
        if self.sweep or self.use_wandb:
            run_id = self.wandb_init()
            self.output_dir = self.output_dir / run_id
            self.output_dir.mkdir(exist_ok=True)

        # Load info for continuing from loss_dict
        if self.cont:
            print("Continuing run, checking for loss_dict.json", flush=True)
            # Try and load loss_dict
            loss_dict_fn = self.output_dir / "loss_dict.json"
            if loss_dict_fn.exists():
                print("Found loss_dict.json", flush=True)
                self.loss_dict = json.loads(loss_dict_fn.read_text())
                self.start_epoch = len(
                    next(iter(self.loss_dict["train"].values()))["losses"]
                )
            else:
                print("No loss_dict file found.")
                if self.log_file:
                    self.logger.info("No loss_dict file found.")

                all_weights_epochs = [
                    int(p.stem)
                    for p in self.output_dir.glob("*.th")
                    if p.stem.isnumeric()
                ]
                if len(all_weights_epochs) == 0:
                    raise FileNotFoundError("No weights files found from previous run.")

                self.start_epoch = max(all_weights_epochs) + 1

            # Get splits directly from loss_dict, if available (otherwise will fall
            #  back to splits from self.ds_splitter_config)
            if len(self.loss_dict) > 0:
                subset_idxs = {"train": [], "val": [], "test": []}
                for i, (compound, _) in enumerate(self.ds):
                    if self.model_config.grouped:
                        compound_id = compound
                    else:
                        compound_id = compound[1]

                    for sp in ["train", "val", "test"]:
                        if compound_id in self.loss_dict[sp]:
                            print(f"putting {compound_id} in {sp}", flush=True)
                            subset_idxs[sp].append(i)
                            break
                    else:
                        print(
                            (
                                f"Compound {compound_id} not found in loss_dict, not "
                                "including it."
                            )
                        )
                        if self.log_file:
                            self.logger.info(
                                (
                                    f"Compound {compound_id} not found in loss_dict, not "
                                    "including it."
                                )
                            )

                self.ds_train = torch.utils.data.Subset(self.ds, subset_idxs["train"])
                self.ds_val = torch.utils.data.Subset(self.ds, subset_idxs["val"])
                self.ds_test = torch.utils.data.Subset(self.ds, subset_idxs["test"])

            # Load model weights
            try:
                weights_path = self.output_dir / f"{self.start_epoch - 1}.th"
                self.model_config = self.model_config.update(
                    {
                        "model_weights": torch.load(
                            weights_path, map_location=self.device
                        )
                    }
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Found {self.start_epoch} epochs of training, but didn't find "
                    f"{self.start_epoch - 1}.th weights file."
                )

        print(
            "ds lengths",
            len(self.ds_train),
            len(self.ds_val),
            len(self.ds_test),
            flush=True,
        )

        # Build the Model
        self.model = self.model_config.build().to(self.device)

        # Build the Optimizer
        self.optimizer = self.optimizer_config.build(self.model.parameters())

        # Load optimizer state for continuing
        if self.cont:
            optimizer_state_fn = self.output_dir / "optimizer.th"
            if not optimizer_state_fn.exists():
                raise FileNotFoundError("No optimizer state file found.")

            optimizer_state = torch.load(optimizer_state_fn, map_location=self.device)
            self.optimizer.load_state_dict(optimizer_state)

        # Build early stopping
        if self.es_config:
            self.es = self.es_config.build()
        else:
            self.es = None

        # Build loss function
        self.loss_func = self.loss_config.build()

        # Set internal tracker to True so we know we can start training
        self._is_initialized = True

    def train(self):
        """
        Train the model, updating the model and loss_dict in-place.
        """
        if not self._is_initialized:
            if self.auto_init:
                self.initialize()
            else:
                raise ValueError("Trainer was not initialized before trying to train.")

        # Save initial model weights for debugging
        if not self.cont:
            torch.save(self.model.state_dict(), self.output_dir / "init.th")

        # Train for n epochs
        for epoch_idx in range(self.start_epoch, self.n_epochs):
            print(f"Epoch {epoch_idx}/{self.n_epochs}", flush=True)
            if self.log_file:
                self.logger.info(f"Epoch {epoch_idx}/{self.n_epochs}")
            if epoch_idx % 10 == 0 and epoch_idx > 0:
                train_loss = np.mean(
                    [v["losses"][-1] for v in self.loss_dict["train"].values()]
                )
                val_loss = np.mean(
                    [v["losses"][-1] for v in self.loss_dict["val"].values()]
                )
                test_loss = np.mean(
                    [v["losses"][-1] for v in self.loss_dict["test"].values()]
                )
                print(f"Training loss: {train_loss:0.5f}")
                print(f"Validation loss: {val_loss:0.5f}")
                print(f"Testing loss: {test_loss:0.5f}", flush=True)
                if self.log_file:
                    self.logger.info(f"Training loss: {train_loss:0.5f}")
                    self.logger.info(f"Validation loss: {val_loss:0.5f}")
                    self.logger.info(f"Testing loss: {test_loss:0.5f}")
            tmp_loss = []

            # Initialize batch
            batch_counter = 0
            self.optimizer.zero_grad()
            start_time = time()
            for compound, pose in self.ds_train:
                if type(compound) is tuple:
                    compound_id = compound[1]
                else:
                    compound_id = compound

                # convert to float to match other types
                target = torch.tensor(
                    [[pose[self.target_prop]]], device=self.device
                ).float()
                in_range = torch.tensor(
                    [[pose[f"{self.target_prop}_range"]]], device=self.device
                ).float()
                uncertainty = torch.tensor(
                    [[pose[f"{self.target_prop}_stderr"]]],
                    device=self.device,
                ).float()

                # Get input poses for GroupedModel
                if self.model_config.grouped:
                    model_inp = pose["poses"]
                else:
                    model_inp = pose

                # Make prediction and calculate loss
                pred, pose_preds = self.model(model_inp)
                pred = pred.reshape(target.shape)
                pose_preds = [p.item() for p in pose_preds]
                loss = self.loss_func(pred, target, in_range, uncertainty)

                # Can just call loss.backward, grads will accumulate additively
                loss.backward()

                # Update loss_dict
                self._update_loss_dict(
                    "train",
                    compound_id,
                    target.item(),
                    in_range.item(),
                    uncertainty.item(),
                    pred.item(),
                    loss.item(),
                    pose_preds=pose_preds,
                )

                # Keep track of loss for each sample
                tmp_loss.append(loss.item())

                batch_counter += 1

                # Perform backprop if we've done all the preds for this batch
                if batch_counter == self.batch_size:
                    # Backprop
                    self.optimizer.step()
                    if any(
                        [
                            p.grad.isnan().any().item()
                            for p in self.model.parameters()
                            if p.grad is not None
                        ]
                    ):
                        raise ValueError("NaN gradients")

                    # Reset batch tracking
                    batch_counter = 0
                    self.optimizer.zero_grad()

            if batch_counter > 0:
                # Backprop for final incomplete batch
                self.optimizer.step()
                if any(
                    [
                        p.grad.isnan().any().item()
                        for p in self.model.parameters()
                        if p.grad is not None
                    ]
                ):
                    raise ValueError("NaN gradients")
            end_time = time()

            epoch_train_loss = np.mean(tmp_loss)

            self.model.eval()
            tmp_loss = []
            for compound, pose in self.ds_val:
                if type(compound) is tuple:
                    compound_id = compound[1]
                else:
                    compound_id = compound

                # convert to float to match other types
                target = torch.tensor(
                    [[pose[self.target_prop]]], device=self.device
                ).float()
                in_range = torch.tensor(
                    [[pose[f"{self.target_prop}_range"]]], device=self.device
                ).float()
                uncertainty = torch.tensor(
                    [[pose[f"{self.target_prop}_stderr"]]],
                    device=self.device,
                ).float()

                # Get input poses for GroupedModel
                if self.model_config.grouped:
                    model_inp = pose["poses"]
                else:
                    model_inp = pose

                # Make prediction and calculate loss
                pred, pose_preds = self.model(model_inp)
                pred = pred.reshape(target.shape)
                pose_preds = [p.item() for p in pose_preds]
                loss = self.loss_func(pred, target, in_range, uncertainty)

                # Update loss_dict
                self._update_loss_dict(
                    "val",
                    compound_id,
                    target.item(),
                    in_range.item(),
                    uncertainty.item(),
                    pred.item(),
                    loss.item(),
                    pose_preds=pose_preds,
                )

                tmp_loss.append(loss.item())
            epoch_val_loss = np.mean(tmp_loss)

            tmp_loss = []
            for compound, pose in self.ds_test:
                if type(compound) is tuple:
                    compound_id = compound[1]
                else:
                    compound_id = compound

                # convert to float to match other types
                target = torch.tensor(
                    [[pose[self.target_prop]]], device=self.device
                ).float()
                in_range = torch.tensor(
                    [[pose[f"{self.target_prop}_range"]]], device=self.device
                ).float()
                uncertainty = torch.tensor(
                    [[pose[f"{self.target_prop}_stderr"]]],
                    device=self.device,
                ).float()

                # Get input poses for GroupedModel
                if self.model_config.grouped:
                    model_inp = pose["poses"]
                else:
                    model_inp = pose

                # Make prediction and calculate loss
                pred, pose_preds = self.model(model_inp)
                pred = pred.reshape(target.shape)
                pose_preds = [p.item() for p in pose_preds]
                loss = self.loss_func(pred, target, in_range, uncertainty)

                # Update loss_dict
                self._update_loss_dict(
                    "test",
                    compound_id,
                    target.item(),
                    in_range.item(),
                    uncertainty.item(),
                    pred.item(),
                    loss.item(),
                    pose_preds=pose_preds,
                )

                tmp_loss.append(loss.item())
            epoch_test_loss = np.mean(tmp_loss)
            self.model.train()

            if self.use_wandb or self.sweep:
                wandb.log(
                    {
                        "train_loss": epoch_train_loss,
                        "val_loss": epoch_val_loss,
                        "test_loss": epoch_test_loss,
                        "epoch": epoch_idx,
                        "epoch_time": end_time - start_time,
                    }
                )
            # Save states
            torch.save(self.model.state_dict(), self.output_dir / f"{epoch_idx}.th")
            torch.save(self.optimizer.state_dict(), self.output_dir / "optimizer.th")
            (self.output_dir / "loss_dict.json").write_text(json.dumps(self.loss_dict))

            # Stop if loss has gone to infinity or is NaN
            if (
                np.isnan(epoch_train_loss)
                or (epoch_train_loss == np.inf)
                or (epoch_train_loss == -np.inf)
            ):
                (self.output_dir / "ds_train.pkl").write_bytes(pkl.dumps(self.ds_train))
                (self.output_dir / "ds_val.pkl").write_bytes(pkl.dumps(self.ds_val))
                (self.output_dir / "ds_test.pkl").write_bytes(pkl.dumps(self.ds_test))
                raise ValueError("Unrecoverable loss value reached.")

            # Stop training if EarlyStopping says to
            if self.es:
                if self.es_config.es_type == "best" and self.es.check(
                    epoch_idx, epoch_val_loss, self.model.state_dict()
                ):
                    print(
                        (
                            f"Stopping training after epoch {epoch_idx}, "
                            f"using weights from epoch {self.es.best_epoch}"
                        ),
                        flush=True,
                    )
                    if self.log_file:
                        self.logger.info(
                            f"Stopping training after epoch {epoch_idx}, "
                            f"using weights from epoch {self.es.best_epoch}"
                        )
                    self.model.load_state_dict(self.es.best_wts)
                    if self.use_wandb or self.sweep:
                        wandb.log(
                            {
                                "best_epoch": self.es.best_epoch,
                                "best_loss": self.es.best_loss,
                            }
                        )
                    use_epoch = self.es.best_epoch
                    break
                elif self.es_config.es_type == "patient_converged" and self.es.check(
                    epoch_idx, epoch_val_loss, self.model.state_dict()
                ):
                    print(
                        (
                            f"Stopping training after epoch {epoch_idx}, "
                            f"using weights from epoch {self.es.converged_epoch}"
                        ),
                        flush=True,
                    )
                    if self.log_file:
                        self.logger.info(
                            f"Stopping training after epoch {epoch_idx}, "
                            f"using weights from epoch {self.es.converged_epoch}"
                        )
                    self.model.load_state_dict(self.es.converged_wts)
                    if self.use_wandb or self.sweep:
                        wandb.log(
                            {
                                "converged_epoch": self.es.converged_epoch,
                                "converged_loss": self.es.converged_loss,
                            }
                        )
                    use_epoch = self.es.converged_epoch
                    break
                elif self.es_config.es_type == "converged" and self.es.check(
                    epoch_val_loss
                ):
                    print(f"Stopping training after epoch {epoch_idx}", flush=True)
                    if self.log_file:
                        self.logger.info(f"Stopping training after epoch {epoch_idx}")
                    use_epoch = epoch_idx
                    break
        else:
            use_epoch = None

        if use_epoch is not None:
            (self.output_dir / "loss_dict_full.json").write_text(
                json.dumps(self.loss_dict)
            )
            # Trim the loss_dict
            self.loss_dict = {
                sp: {
                    compound_id: {
                        k: v[: use_epoch + 1] if isinstance(v, list) else v
                        for k, v in compound_d.items()
                    }
                    for compound_id, compound_d in sp_d.items()
                }
                for sp, sp_d in self.loss_dict.items()
            }

        if self.use_wandb or self.sweep:
            wandb.finish()

        torch.save(self.model.state_dict(), self.output_dir / "final.th")
        (self.output_dir / "loss_dict.json").write_text(json.dumps(self.loss_dict))

    def _update_loss_dict(
        self,
        split,
        compound_id,
        target,
        in_range,
        uncertainty,
        pred,
        loss,
        pose_preds=None,
    ):
        """
        Update (in-place) loss_dict info from training/evaluation on a molecule.

        Parameters
        ----------
        split : str
            Which split ["train", "val", "test"]
        compound_id : str
            Compound ID
        target : float
            Target value for this compound
        in_range : int
            Whether target is below (-1), within (0), or above (1) the assay range
        uncertainty : float
            Experimental measurement uncertainty
        pred : float
           Model prediction
        loss : float
            Prediction loss
        pose_preds : float, optional
            Single-pose model prediction for each pose in input (for multi-pose models)
        """
        if split not in self.loss_dict:
            self.loss_dict[split] = {}

        if compound_id in self.loss_dict[split]:
            self.loss_dict[split][compound_id]["preds"].append(pred)
            if pose_preds is not None:
                self.loss_dict[split][compound_id]["pose_preds"].append(pose_preds)
            self.loss_dict[split][compound_id]["losses"].append(loss)
        else:
            self.loss_dict[split][compound_id] = {
                "target": target,
                "in_range": in_range,
                "uncertainty": uncertainty,
                "preds": [pred],
                "losses": [loss],
            }
            if pose_preds is not None:
                self.loss_dict[split][compound_id]["pose_preds"] = [pose_preds]

    def _make_wandb_ds_tables(self):
        ds_tables = []

        for ds in [self.ds_train, self.ds_val, self.ds_test]:
            table = wandb.Table(
                columns=[
                    "crystal",
                    "compound_id",
                    self.target_prop,
                    f"{self.target_prop}_range",
                    f"{self.target_prop}_stderr",
                    "date_created",
                ]
            )
            # Build table and add each molecule
            for compound, d in ds:
                try:
                    # This should work for all structural datasets
                    xtal_id, compound_id = d["compound"]
                except KeyError:
                    # This should only trigger for graph datasets
                    xtal_id = ""
                    compound_id = compound

                try:
                    target_value = d[self.target_prop]
                except KeyError:
                    target_value = np.nan
                try:
                    target_value_range = d[f"{self.target_prop}_range"]
                except KeyError:
                    target_value_range = np.nan
                try:
                    target_value_stderr = d[f"{self.target_prop}_stderr"]
                except KeyError:
                    target_value_stderr = np.nan
                except AttributeError:
                    target_value = d[self.target_prop]
                try:
                    date_created = d["date_created"]
                except KeyError:
                    date_created = None
                table.add_data(
                    xtal_id,
                    compound_id,
                    target_value,
                    target_value_range,
                    target_value_stderr,
                    date_created,
                )

            ds_tables.append(table)

        return ds_tables
