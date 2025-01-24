import json
import pickle as pkl
from copy import deepcopy
from glob import glob
from pathlib import Path
from time import time

import numpy as np
import torch
import wandb
from asapdiscovery.data.services.aws.s3 import S3
from asapdiscovery.data.services.services_config import S3Settings
from asapdiscovery.data.util.logging import FileLogger
from asapdiscovery.ml.config import (
    DataAugConfig,
    DatasetConfig,
    DatasetSplitterConfig,
    EarlyStoppingConfig,
    LossFunctionConfig,
    OptimizerConfig,
)
from asapdiscovery.ml.dataset import dataset_to_csv
from asapdiscovery.ml.schema import TrainingPredictionTracker
from mtenn.config import (
    E3NNModelConfig,
    GATModelConfig,
    ModelConfigBase,
    ModelType,
    SchNetModelConfig,
    ViSNetModelConfig,
)
from pydantic.v1 import (
    BaseModel,
    Extra,
    Field,
    ValidationError,
    confloat,
    conlist,
    root_validator,
    validator,
)


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
    es_config: EarlyStoppingConfig = Field(
        ..., description="Config describing the early stopping check to use."
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
    loss_configs: conlist(item_type=LossFunctionConfig, min_items=1) = Field(
        ...,
        description="Config describing the loss function for training.",
    )
    loss_weights: torch.Tensor = Field(
        [],
        description=(
            "Weight for each loss function. Values will be normalized to add up to 1. "
            "If no values are passed, each loss function will be weighted equally. If "
            "any values are passed, there must be one for each loss function."
        ),
    )
    eval_loss_weights: torch.Tensor = Field(
        [],
        description=(
            "Weight for each loss function when calculating val and test losses. "
            "Values will be normalized to add up to 1. If no values are passed, the "
            "weights from loss_weights will be used. If any values are passed, there "
            "must be one for each loss function."
        ),
    )
    weight_decay: confloat(ge=0.0, allow_inf_nan=False) = Field(
        0.0,
        description=(
            "Weight decay weighting for training. This will add a term of "
            "weight_decay / 2 * the square of the L2-norm of the model weights, "
            "excluding any bias terms."
        ),
    )
    batch_norm: bool = Field(
        False, description="Normalize batch gradient by batch size."
    )
    data_aug_configs: list[DataAugConfig] = Field(
        [],
        description="List of data augmentations to be applied in order to each pose.",
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
        description=(
            "Which epoch to start training at (used for continuing training runs)."
        ),
    )
    n_epochs: int = Field(
        300,
        description=(
            "Which epoch to stop training at. For non-continuation runs, this "
            "will be the total number of epochs to train for."
        ),
    )
    batch_size: int = Field(
        -1,
        description=(
            "Number of samples to predict on before performing backprop."
            "Set to -1 (default) to use the entire training set as a batch."
        ),
    )
    target_prop: str = Field("pIC50", description=("Target property to train against."))
    cont: bool = Field(
        False, description="This is a continuation of a previous training run."
    )
    pred_tracker: TrainingPredictionTracker = Field(
        None,
        description=(
            "TrainingPredictionTracker to keep track of predictions and losses over "
            "training."
        ),
    )
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
    save_weights: str = Field(
        "all",
        description=(
            "How often to save weights during training."
            'Options are to keep every epoch ("all"), only keep the most recent '
            'epoch ("recent"), or only keep the final epoch ("final").'
        ),
    )

    write_ds_csv: bool = Field(
        False, description="Write the dataset splits to CSV files."
    )

    # W&B parameters
    use_wandb: bool = Field(False, description="Use W&B to log model training.")
    wandb_project: str | None = Field(None, description="W&B project name.")
    wandb_name: str | None = Field(None, description="W&B project name.")
    wandb_group: str | None = Field(None, description="W&B group name.")
    extra_config: dict | None = Field(
        None, description="Any extra config options to log to W&B."
    )
    wandb_run_id: str | None = Field(None, description="W&B run ID.")

    # artifact tracking options
    upload_to_s3: bool = Field(False, description="Upload artifacts to S3.")
    s3_settings: S3Settings | None = Field(None, description="S3 settings.")
    s3_path: str | None = Field(None, description="S3 location to upload artifacts.")
    model_tag: str | None = Field(None, description="Tag for the model being trained.")

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
            "loss_funcs": {"exclude": True},
        }

        # Allow things to be added to the object after initialization/validation
        extra = Extra.allow

        # Custom encoder to cast device to str before trying to serialize
        json_encoders = {
            torch.device: lambda d: str(d),
            torch.Tensor: lambda t: t.tolist(),
        }

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

    @validator("data_aug_configs", "loss_configs", pre=True)
    def load_cache_files_lists(cls, kwargs_list, field):
        """
        This validator performs the same functionality as the above function, but for
        Fields that contain a list of some type.
        """
        config_cls = field.type_

        if isinstance(kwargs_list, dict):
            # This will occur in the even of a Sweep, in which case the values will be
            #  a dict mapping index in the list to a value
            # Just need to extract the values in the correct order (cast indices to int
            #  just in case)
            kwargs_list = [kwargs_list[i] for i in sorted(kwargs_list, key=int)]

        configs = []
        for config_kwargs in kwargs_list:
            # If an instance of the actual config class is passed, there's no cache file so
            #  just return
            if isinstance(config_kwargs, config_cls):
                configs.append(config_kwargs)
                continue

            # Just skip any Nones
            if config_kwargs is None:
                continue

            if isinstance(config_kwargs, str):
                if len(config_kwargs) == 0:
                    continue

                # Parse into dict
                config_kwargs = dict(
                    [kvp.split(":") for kvp in config_kwargs.split(",")]
                )

            # Get config cache file and overwrite option (if given). Defaults to no cache
            #  file and not overwriting
            config_file = config_kwargs.pop("cache", None)
            overwrite = config_kwargs.pop("overwrite_cache", False)

            configs.append(
                Trainer._build_arbitrary_config(
                    config_cls=config_cls,
                    config_file=config_file,
                    overwrite=overwrite,
                    **config_kwargs,
                )
            )

        return configs

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
                key, val = kvp.split(",")
            except Exception:
                raise ValueError(f"Couldn't parse key,value pair '{kvp}'.")

            extra_config[key] = val

        return extra_config

    @validator("loss_weights", pre=True, always=True)
    def check_loss_weights(cls, v, values):
        """
        Make sure that we have the right number of loss function weights, and cast to
        normalized tensor.
        """
        if (len(v) > 0) and (len(v) != len(values["loss_configs"])):
            raise ValueError(
                f"Mismatch between number of loss function weights ({len(v)}) and "
                f"number of loss functions ({len(values['loss_configs'])})."
            )

        # Fill with 1s if no values passed
        if len(v) == 0:
            v = [1] * len(values["loss_configs"])
        elif isinstance(v, dict):
            # This will occur in the even of a Sweep, in which case the values will be
            #  a dict mapping index in the list to a value
            # Just need to extract the values in the correct order (cast indices to int
            #  just in case)
            v = [v[i] for i in sorted(v, key=int)]

        # Cast to tensor (don't send to device in case we're building the Trainer on a
        #  CPU-only node)
        v = torch.tensor(v, dtype=torch.float32)

        # Check for negative numbers
        if (v < 0).any():
            raise ValueError("Values for loss_weights can't be negative.")

        # Normalize to 1
        v /= v.sum()

        return v

    @validator("eval_loss_weights", pre=True, always=True)
    def check_eval_loss_weights(cls, v, values):
        """
        Make sure that we have the right number of loss function weights, and cast to
        normalized tensor.
        """
        if (len(v) > 0) and (len(v) != len(values["loss_configs"])):
            raise ValueError(
                f"Mismatch between number of loss function weights ({len(v)}) and "
                f"number of loss functions ({len(values['loss_configs'])})."
            )

        # Fill with 1s if no values passed
        if len(v) == 0:
            return values["loss_weights"]
        elif isinstance(v, dict):
            # This will occur in the even of a Sweep, in which case the values will be
            #  a dict mapping index in the list to a value
            # Just need to extract the values in the correct order (cast indices to int
            #  just in case)
            v = [v[i] for i in sorted(v, key=int)]

        # Cast to tensor (don't send to device in case we're building the Trainer on a
        #  CPU-only node)
        v = torch.tensor(v, dtype=torch.float32)

        # Check for negative numbers
        if (v < 0).any():
            raise ValueError("Values for loss_weights can't be negative.")

        # Normalize to 1
        v /= v.sum()

        return v

    @validator("pred_tracker", always=True)
    def init_pred_tracker(cls, pred_tracker):
        # If a value was passed, it's already been validated so just return that
        if isinstance(pred_tracker, TrainingPredictionTracker):
            return pred_tracker
        else:
            # Otherwise need to init an empty one
            return TrainingPredictionTracker()

    @validator("save_weights")
    def check_save_weights(cls, v):
        """
        Just make sure the option is one of the valid ones.
        """
        v = v.lower()

        if v not in {"all", "recent", "final"}:
            raise ValueError(f'Invalid option for save_weights: "{v}"')

        return v

    @root_validator
    def check_s3_settings(cls, values):
        """
        check that if we uploading to S3 that the S3 path is set
        """
        upload_to_s3 = values.get("upload_to_s3")
        s3_path = values.get("s3_path")
        if upload_to_s3 and not s3_path:
            raise ValueError("Must provide an S3 path if uploading to S3.")
        return values

    @validator("s3_path", pre=True)
    def check_s3_path(cls, v):
        # check it is a folder path not a file path, cast to Path
        if v:
            if Path(v).suffix:
                raise ValueError("S3 path must be a folder path.")
        return v

    def wandb_init(self):
        """
        Initialize WandB, handling saving the run ID (for continuing the run later).

        Returns
        -------
        str
            The WandB run ID for the initialized run
        """

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
                raise FileNotFoundError("Couldn't find run_id file to continue run.")
            # Make sure the run_id is legit
            try:
                wandb.init(project=self.wandb_project, id=run_id, resume="must")
            except wandb.errors.UsageError:
                raise wandb.errors.UsageError(
                    f"Run in run_id file ({run_id}) doesn't exist"
                )
            self.wandb_run_id = run_id
            # Update run config to reflect it's been resumed
            wandb.config.update(config, allow_val_change=True)
        else:
            # Start new run
            run_id = wandb.init(
                project=self.wandb_project,
                config=config,
                name=self.wandb_name,
                group=self.wandb_group,
            ).id
            self.wandb_run_id = run_id

            # Save run_id in case we want to continue later
            if not self.output_dir.exists():
                print(
                    "No output directory specified, not saving run_id anywhere.",
                    flush=True,
                )
            else:
                run_id_fn.write_text(run_id)

        for split, table in zip(["train", "val", "test"], self._make_wandb_ds_tables()):
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

        # check S3 settings so that fail early if there is an issue
        if self.upload_to_s3:
            if self.s3_settings is None:
                try:
                    self.s3_settings = S3Settings()
                except Exception as e:
                    raise ValueError(f"Error loading S3 settings: {e}")

        # Build dataset and split
        self.ds = self.ds_config.build()
        self.ds_train, self.ds_val, self.ds_test = self.ds_splitter_config.split(
            self.ds
        )

        # Adjust output_dir and make sure it exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Start the W&B process
        if self.use_wandb:
            run_id = self.wandb_init()
            self.output_dir = self.output_dir / run_id
            self.output_dir.mkdir(exist_ok=True)

        # Load info for continuing from pred_tracker
        if self.cont:
            print("Continuing run, checking for pred_tracker.json", flush=True)
            # Try and load pred_tracker
            pred_tracker_fn = self.output_dir / "pred_tracker.json"
            if pred_tracker_fn.exists():
                print("Found pred_tracker.json", flush=True)
                self.pred_tracker = TrainingPredictionTracker(
                    **json.loads(pred_tracker_fn.read_text())
                )
                try:
                    self.start_epoch = len(next(iter(self.pred_tracker))[1].predictions)
                except StopIteration:
                    self.start_epoch = 0
            else:
                print("No pred_tracker file found.")
                if self.log_file:
                    self.logger.info("No pred_tracker file found.")

                all_weights_epochs = [
                    int(p.stem)
                    for p in self.output_dir.glob("*.th")
                    if p.stem.isnumeric()
                ]
                if len(all_weights_epochs) == 0:
                    raise FileNotFoundError("No weights files found from previous run.")

                self.start_epoch = max(all_weights_epochs) + 1

            # Get splits directly from pred_tracker, if available (otherwise will fall
            #  back to splits from self.ds_splitter_config)
            if len(self.pred_tracker) > 0:
                subset_idxs = {"train": [], "val": [], "test": []}

                # First build a dict mapping compound_id: idx in ds
                compound_idx_dict = {}
                for i, (compound, _) in enumerate(self.ds):
                    if self.model_config.grouped:
                        compound_id = compound
                    else:
                        compound_id = compound[1]
                    if compound_id in compound_idx_dict:
                        raise ValueError(
                            f"Found multiple entries in ds for compound {compound_id}"
                        )
                    compound_idx_dict[compound_id] = i

                for _, tp in self.pred_tracker:
                    if tp.compound_id not in compound_idx_dict:
                        raise ValueError(
                            f"Found compound {tp.compound_id} in pred_tracker "
                            "but not in ds"
                        )

                subset_idxs = {
                    sp: [
                        compound_idx_dict[compound_id]
                        for compound_id in compound_id_list
                    ]
                    for sp, compound_id_list in self.pred_tracker.get_compound_ids().items()
                }

                self.ds_train = torch.utils.data.Subset(self.ds, subset_idxs["train"])
                self.ds_val = torch.utils.data.Subset(self.ds, subset_idxs["val"])
                self.ds_test = torch.utils.data.Subset(self.ds, subset_idxs["test"])

            # Load model weights
            try:
                weights_path = self.output_dir / f"{self.start_epoch - 1}.th"
                if not weights_path.exists():
                    weights_path = self.output_dir / "weights.th"
                print(f"Using weights file {weights_path.name}", flush=True)
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
                    f"{self.start_epoch - 1}.th weights file or weights.th weights "
                    "file."
                )

        print(
            "ds lengths",
            len(self.ds_train),
            len(self.ds_val),
            len(self.ds_test),
            flush=True,
        )

        # write the datasets to CSV
        if self.write_ds_csv:
            dataset_to_csv(self.ds_train, self.output_dir / "ds_train.csv")
            dataset_to_csv(self.ds_val, self.output_dir / "ds_val.csv")
            dataset_to_csv(self.ds_test, self.output_dir / "ds_test.csv")

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
        self.es = self.es_config.build()

        # Build data augmentation classes
        self.data_augs = [aug.build() for aug in self.data_aug_configs]

        # Build loss function
        self.loss_funcs = [loss.build() for loss in self.loss_configs]

        # Send loss_weights to appropriate device
        self.loss_weights = self.loss_weights.to(self.device)
        self.eval_loss_weights = self.eval_loss_weights.to(self.device)

        # Set internal tracker to True so we know we can start training
        self._is_initialized = True

    def train(self):
        """
        Train the model, updating the model and pred_tracker in-place.
        """
        if not self._is_initialized:
            if self.auto_init:
                self.initialize()
            else:
                raise ValueError("Trainer was not initialized before trying to train.")

        # Save initial model weights for debugging
        if not self.cont:
            torch.save(self.model.state_dict(), self.output_dir / "init.th")

        # Return early if trying to continue but training has already reached the end
        if self.start_epoch == self.n_epochs:
            print("Alrady trained for all epochs, not resuming training.", flush=True)
            if self.use_wandb:
                wandb.finish()
            return

        # Train for n epochs
        for epoch_idx in range(self.start_epoch, self.n_epochs):
            print(f"Epoch {epoch_idx}/{self.n_epochs}", flush=True)
            if self.log_file:
                self.logger.info(f"Epoch {epoch_idx}/{self.n_epochs}")
            if epoch_idx % 10 == 0 and epoch_idx > 0:
                train_loss = np.mean(
                    [v.loss_vals[-1] for v in self.pred_tracker.split_dict["train"]]
                )
                val_loss = np.mean(
                    [v.loss_vals[-1] for v in self.pred_tracker.split_dict["val"]]
                )
                test_loss = np.mean(
                    [v.loss_vals[-1] for v in self.pred_tracker.split_dict["test"]]
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
                    xtal_id, compound_id = compound
                else:
                    xtal_id = "NA"
                    compound_id = compound

                try:
                    # convert to float to match other types
                    target = torch.tensor(
                        pose[self.target_prop], device=self.device
                    ).float()
                except KeyError:
                    print(
                        f"{self.target_prop} not found in compound {compound}, skipping.",
                        flush=True,
                    )
                    if self.log_file:
                        self.logger.info(
                            f"{self.target_prop} not found in compound {compound}, skipping."
                        )
                    continue
                in_range = (
                    torch.tensor(
                        pose[f"{self.target_prop}_range"], device=self.device
                    ).float()
                    if f"{self.target_prop}_range" in pose
                    else None
                )
                uncertainty = (
                    torch.tensor(
                        pose[f"{self.target_prop}_stderr"], device=self.device
                    ).float()
                    if f"{self.target_prop}_range" in pose
                    else None
                )

                # Get input poses for GroupedModel
                if self.model_config.grouped:
                    model_inp = []
                    for single_pose in pose["poses"]:
                        # Apply all data augmentations
                        aug_pose = deepcopy(single_pose)
                        for aug in self.data_augs:
                            aug_pose = aug(aug_pose)

                        model_inp.append(aug_pose)

                else:
                    # Apply all data augmentations
                    aug_pose = deepcopy(pose)
                    for aug in self.data_augs:
                        aug_pose = aug(aug_pose)

                    model_inp = aug_pose

                # Make prediction and calculate loss
                pred, pose_preds = self.model(model_inp)

                losses = [
                    (
                        loss_func(
                            pred, pose_preds, target, in_range, uncertainty
                        ).reshape((1,))
                    )
                    for loss_func in self.loss_funcs
                ]
                losses = torch.cat(
                    [loss.to(self.device, dtype=torch.float32) for loss in losses]
                )

                # Calculate final loss based on loss weights
                loss = losses.flatten().dot(self.loss_weights)

                # Update pred_tracker
                for (
                    loss_val,
                    loss_config,
                    loss_wt,
                ) in zip(
                    losses,
                    self.loss_configs,
                    self.loss_weights,
                ):
                    if target is None:
                        continue
                    self.pred_tracker.update_values(
                        prediction=pred.item(),
                        pose_predictions=[p.item() for p in pose_preds],
                        loss_val=loss_val.item(),
                        split="train",
                        compound_id=compound_id,
                        xtal_id=xtal_id,
                        target_prop=self.target_prop,
                        target_val=target,
                        in_range=in_range,
                        uncertainty=uncertainty,
                        loss_config=loss_config,
                        loss_weight=loss_wt,
                    )

                # If all target props were missing, there's no backprop to do
                if not loss.requires_grad:
                    continue

                # Add in weight decay term if requested
                if self.weight_decay:
                    # Square of the sum of L2 norms of each parameter (excluding bias
                    #  terms)
                    weight_norm = torch.sum(
                        torch.pow(
                            torch.stack(
                                [
                                    torch.linalg.norm(x)
                                    for n, x in self.model.named_parameters()
                                    if n.split(".")[-1] != "bias"
                                ]
                            ),
                            2,
                        )
                    )
                    loss += self.weight_decay / 2 * weight_norm

                # Can just call loss.backward, grads will accumulate additively
                loss.backward()

                # Keep track of loss for each sample
                tmp_loss.append(loss.item())

                batch_counter += 1

                # Perform backprop if we've done all the preds for this batch
                if batch_counter == self.batch_size:
                    if self.batch_norm:
                        # Need to scale the gradients by batch_size to get to MSE loss
                        for p in self.model.parameters():
                            p.grad /= batch_counter

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
                if self.batch_norm:
                    # Need to scale the gradients by batch_size to get to MSE loss
                    for p in self.model.parameters():
                        p.grad /= batch_counter

                # Backprop for final incomplete batch
                if any(
                    [
                        p.grad.isnan().any().item()
                        for p in self.model.parameters()
                        if p.grad is not None
                    ]
                ):
                    raise ValueError("NaN gradients")

                # Backprop
                self.optimizer.step()

            end_time = time()

            epoch_train_loss = np.mean(tmp_loss)

            # Val and test splits
            self.model.eval()
            tmp_loss = []
            for compound, pose in self.ds_val:
                if type(compound) is tuple:
                    xtal_id, compound_id = compound
                else:
                    xtal_id = "NA"
                    compound_id = compound

                try:
                    # convert to float to match other types
                    target = torch.tensor(
                        pose[self.target_prop], device=self.device
                    ).float()
                except KeyError:
                    print(
                        f"{self.target_prop} not found in compound {compound}, skipping.",
                        flush=True,
                    )
                    if self.log_file:
                        self.logger.info(
                            f"{self.target_prop} not found in compound {compound}, skipping."
                        )
                    continue
                in_range = (
                    torch.tensor(
                        pose[f"{self.target_prop}_range"], device=self.device
                    ).float()
                    if f"{self.target_prop}_range" in pose
                    else None
                )
                uncertainty = (
                    torch.tensor(
                        pose[f"{self.target_prop}_stderr"], device=self.device
                    ).float()
                    if f"{self.target_prop}_range" in pose
                    else None
                )

                # Get input poses for GroupedModel
                if self.model_config.grouped:
                    model_inp = pose["poses"]
                else:
                    model_inp = pose

                # Make prediction and calculate loss
                with torch.no_grad():
                    pred, pose_preds = self.model(model_inp)
                losses = [
                    (
                        loss_func(
                            pred, pose_preds, target, in_range, uncertainty
                        ).reshape((1,))
                    )
                    for loss_func in self.loss_funcs
                ]
                losses = torch.cat(
                    [loss.to(self.device, dtype=torch.float32) for loss in losses]
                )

                # Calculate final loss based on loss weights
                loss = losses.flatten().dot(self.eval_loss_weights)

                # Update pred_tracker
                for (
                    loss_val,
                    loss_config,
                    loss_wt,
                ) in zip(
                    losses,
                    self.loss_configs,
                    self.eval_loss_weights,
                ):
                    if target is None:
                        continue
                    self.pred_tracker.update_values(
                        prediction=pred.item(),
                        pose_predictions=[p.item() for p in pose_preds],
                        loss_val=loss_val.item(),
                        split="val",
                        compound_id=compound_id,
                        xtal_id=xtal_id,
                        target_prop=self.target_prop,
                        target_val=target,
                        in_range=in_range,
                        uncertainty=uncertainty,
                        loss_config=loss_config,
                        loss_weight=loss_wt,
                    )

                tmp_loss.append(loss.item())
            epoch_val_loss = np.mean(tmp_loss)

            tmp_loss = []
            for compound, pose in self.ds_test:
                if type(compound) is tuple:
                    xtal_id, compound_id = compound
                else:
                    xtal_id = "NA"
                    compound_id = compound

                try:
                    # convert to float to match other types
                    target = torch.tensor(
                        pose[self.target_prop], device=self.device
                    ).float()
                except KeyError:
                    print(
                        f"{self.target_prop} not found in compound {compound}, skipping.",
                        flush=True,
                    )
                    if self.log_file:
                        self.logger.info(
                            f"{self.target_prop} not found in compound {compound}, skipping."
                        )
                    continue
                in_range = (
                    torch.tensor(
                        pose[f"{self.target_prop}_range"], device=self.device
                    ).float()
                    if f"{self.target_prop}_range" in pose
                    else None
                )
                uncertainty = (
                    torch.tensor(
                        pose[f"{self.target_prop}_stderr"], device=self.device
                    ).float()
                    if f"{self.target_prop}_range" in pose
                    else None
                )

                # Get input poses for GroupedModel
                if self.model_config.grouped:
                    model_inp = pose["poses"]
                else:
                    model_inp = pose

                # Make prediction and calculate loss
                with torch.no_grad():
                    pred, pose_preds = self.model(model_inp)
                losses = [
                    (
                        loss_func(
                            pred, pose_preds, target, in_range, uncertainty
                        ).reshape((1,))
                    )
                    for loss_func in self.loss_funcs
                ]
                losses = torch.cat(
                    [loss.to(self.device, dtype=torch.float32) for loss in losses]
                )

                # Calculate final loss based on loss weights
                loss = losses.flatten().dot(self.eval_loss_weights)

                # Update pred_tracker
                for (
                    loss_val,
                    loss_config,
                    loss_wt,
                ) in zip(
                    losses,
                    self.loss_configs,
                    self.eval_loss_weights,
                ):
                    if target is None:
                        continue
                    self.pred_tracker.update_values(
                        prediction=pred.item(),
                        pose_predictions=[p.item() for p in pose_preds],
                        loss_val=loss_val.item(),
                        split="test",
                        compound_id=compound_id,
                        xtal_id=xtal_id,
                        target_prop=self.target_prop,
                        target_val=target,
                        in_range=in_range,
                        uncertainty=uncertainty,
                        loss_config=loss_config,
                        loss_weight=loss_wt,
                    )

                tmp_loss.append(loss.item())
            epoch_test_loss = np.mean(tmp_loss)
            self.model.train()

            if self.use_wandb:
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
            if self.save_weights == "all":
                torch.save(self.model.state_dict(), self.output_dir / f"{epoch_idx}.th")
                torch.save(
                    self.optimizer.state_dict(), self.output_dir / "optimizer.th"
                )
            elif self.save_weights == "recent":
                torch.save(self.model.state_dict(), self.output_dir / "weights.th")
                torch.save(
                    self.optimizer.state_dict(), self.output_dir / "optimizer.th"
                )
            (self.output_dir / "pred_tracker.json").write_text(self.pred_tracker.json())

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
                    if self.use_wandb:
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
                    if self.use_wandb:
                        wandb.log(
                            {
                                "converged_epoch": self.es.converged_epoch,
                                "converged_loss": self.es.converged_loss,
                            }
                        )
                    use_epoch = self.es.converged_epoch
                    break
                elif self.es_config.es_type == "converged" and self.es.check(
                    epoch_idx, epoch_val_loss
                ):
                    print(f"Stopping training after epoch {epoch_idx}", flush=True)
                    if self.log_file:
                        self.logger.info(f"Stopping training after epoch {epoch_idx}")
                    use_epoch = epoch_idx
                    break
        else:
            use_epoch = None

        if use_epoch is not None:
            (self.output_dir / "pred_tracker_full.json").write_text(
                self.pred_tracker.json()
            )
            # Trim the pred_tracker
            for _, tp in self.pred_tracker:
                tp.predictions = tp.predictions[: use_epoch + 1]
                tp.pose_predictions = tp.pose_predictions[: use_epoch + 1]
                tp.loss_vals = tp.loss_vals[: use_epoch + 1]

        final_model_path = self.output_dir / "final.th"
        torch.save(self.model.state_dict(), final_model_path)
        (self.output_dir / "pred_tracker.json").write_text(self.pred_tracker.json())

        # write to json
        model_config_path = self.output_dir / "model_config.json"
        model_config_path.write_text(self.model_config.json())

        # copy over the final to tagged model if present
        import shutil

        if self.model_tag:
            final_model_path_tagged = self.output_dir / f"{self.model_tag}.th"
            shutil.copy(final_model_path, final_model_path_tagged)
            final_model_path = final_model_path_tagged

        if self.upload_to_s3:
            print("Uploading to S3")
            if self.model_tag:
                s3_final_model_path = self.s3_path + f"/{self.model_tag}.th"
            else:
                s3_final_model_path = self.s3_path + "/model.th"
            s3_config_path = self.s3_path + "/model_config.json"
            s3 = S3.from_settings(self.s3_settings)
            s3.push_file(
                final_model_path,
                location=s3_final_model_path,
                content_type="application/octet-stream",
            )
            s3.push_file(
                model_config_path,
                location=s3_config_path,
                content_type="application/json",
            )
            if self.use_wandb:
                # track S3 artifacts
                print("Linking S3 artifacts to W&B")
                tag = self.model_tag if self.model_tag else "model"
                model_artifact = wandb.Artifact(
                    tag,
                    type="model",
                    description="trained model",
                )
                model_uri = s3.to_uri(s3_final_model_path)
                print(f"URI: {model_uri}")
                model_artifact.add_reference(model_uri)
                wandb.log_artifact(model_artifact)

                config_artifact = wandb.Artifact(
                    "model_config",
                    type="model_config",
                    description="model configuration",
                )
                config_uri = s3.to_uri(s3_config_path)
                print(f"URI: {config_uri}")
                config_artifact.add_reference(config_uri)
                wandb.log_artifact(config_artifact)

        if self.use_wandb:
            wandb.finish()

    def _make_wandb_ds_tables(self):
        ds_tables = []

        table_cols = ["crystal", "compound_id"]
        table_cols += [
            self.target_prop,
            f"{self.target_prop}_range",
            f"{self.target_prop}_stderr",
        ]
        table_cols += ["date_created"]
        for ds in [self.ds_train, self.ds_val, self.ds_test]:
            table = wandb.Table(columns=table_cols)
            # Build table and add each molecule
            for compound, d in ds:
                try:
                    # This should work for all structural datasets
                    xtal_id, compound_id = d["compound"]
                except KeyError:
                    # This should only trigger for graph datasets
                    xtal_id = ""
                    compound_id = compound

                row_data = [xtal_id, compound_id]
                row_data += [d.get(col, np.nan) for col in table_cols[2:-1]]
                row_data += [d.get("date_created", None)]
                row_data = [
                    str(d) if isinstance(d, np.ndarray) else d for d in row_data
                ]
                table.add_data(*row_data)

            ds_tables.append(table)

        return ds_tables
