import json
import pickle as pkl
from pathlib import Path
from time import time
from typing import Callable

import mtenn
import numpy as np
import torch
import wandb
from asapdiscovery.ml.es import BestEarlyStopping, ConvergedEarlyStopping
from asapdiscovery.ml.schema_v2.config import (
    DatasetConfig,
    DatasetSplitterConfig,
    EarlyStoppingConfig,
    ModelConfigBase,
    OptimizerConfig,
)
from pydantic import BaseModel, Field, validator


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
    loss_func: Callable = Field(..., description="Loss function for training.")

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

    # W&B parameters
    use_wandb: bool = Field(False, description="Use W&B to log model training.")
    sweep: bool = Field(False, description="This run is part of a W&B sweep.")
    wandb_project: str | None = Field(None, description="W&B project name.")
    wandb_name: str | None = Field(None, description="W&B project name.")
    extra_config: list[str] | None = Field(
        None,
        description=(
            "Any extra config options to log to W&B, as a list of "
            "comma-separated key:value pairs."
        ),
    )

    # Tracker to make sure the optimizer and ML model are built before trying to train
    _is_initialized = False

    class Config:
        # Temporary fix for now. This is necessary for the asapdiscovery Dataset
        #  classes, but we should probably figure out a workaround eventurally. Probably
        #  best to implement __get_validators__ for the Dataset classes.
        arbitrary_types_allowed = True

        # For now exclude, but would be good to handle custom serialization for these
        #  classes so we can include as much info as possible
        fields = {"dataset": {"exclude": True}, "loss_dict": {"exclude": True}}

    # Validator to make sure that if output_dir exists, it is a directory
    @validator("output_dir")
    def output_dir_check(cls, p):
        if p.exists():
            assert (
                p.isdir()
            ), "If given output_dir already exists, it must be a directory."

        return p

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
                wandb.config.update({"continue": True}, allow_val_change=True)
            else:
                # Start new run
                run_id = wandb.init(
                    project=self.wandb_project,
                    config=self.dict(),
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

        return run_id

    def initialize(self):
        """
        Build the Optimizer and ML Model described by the stored config.
        """

        # Adjust output_dir and make sure it exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Start the W&B process
        if self.sweep or self.use_wandb:
            run_id = self.wandb_init()
            self.output_dir = self.output_dir / run_id
        self.output_dir.mkdir(exist_ok=True)

        # If sweep or continuing a run, get the optimizer and model config options from
        #  the W&B config
        if self.sweep or self.cont:
            wandb_optimizer_config = wandb.config["optimizer_config"]
            wandb_model_config = wandb.config["model_config"]

            self.optimizer_config = self.optimizer_config.update(wandb_optimizer_config)
            self.model_config = self.model_config.update(wandb_model_config)

        # Build the Model
        self.model = self.model_config.build().to(self.device)

        # Build the Optimizer
        self.optimizer = self.optimizer_config.build(self.model.parameters())

        # Build early stopping
        if self.es_config:
            self.es = self.es_config.build()

        # Build dataset and split
        self.ds = self.ds_config.build()
        self.ds_train, self.ds_val, self.ds_test = self.ds_splitter_config.split(
            self.ds
        )

        # Set internal tracker to True so we know we can start training
        self._is_initialized = True

    def train(self) -> (mtenn.model.Model, dict):
        """
        Train the model, returning the trained model and loss_dict.

        Returns
        -------
        mtenn.model.Model
            Trained model
        dict
            Loss dict
        """
        if not self._is_initialized:
            if self.auto_init:
                self.initialize()
            else:
                raise ValueError("Trainer was not initialized before trying to train.")

        # Save initial model weights for debugging
        torch.save(self.model.state_dict(), self.output_dir / "init.th")

        # Train for n epochs
        for epoch_idx in range(self.start_epoch, self.n_epochs):
            print(f"Epoch {epoch_idx}/{self.n_epochs}", flush=True)
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
            tmp_loss = []

            # Initialize batch
            batch_counter = 0
            self.optimizer.zero_grad()
            batch_loss = None
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
                    [[pose["pIC50_range"]]], device=self.device
                ).float()
                uncertainty = torch.tensor(
                    [[pose[f"{self.target_prop}_stderr"]]],
                    device=self.device,
                ).float()

                # Make prediction and calculate loss
                pred, pose_preds = self.model(pose)
                pred = pred.reshape(target.shape)
                pose_preds = [p.item() for p in pose_preds]
                loss = self.loss_func(pred, target, in_range, uncertainty)

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

                # Update batch_loss
                if batch_loss is None:
                    batch_loss = loss
                else:
                    batch_loss += loss
                batch_counter += 1

                # Perform backprop if we've done all the preds for this batch
                if batch_counter == self.batch_size:
                    # Backprop
                    batch_loss.backward()
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
                    batch_loss = None

            if batch_counter > 0:
                # Backprop for final incomplete batch
                batch_loss.backward()
                self.optimizer.step()
                if any([p.grad.isnan().any().item() for p in self.model.parameters()]):
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
                    [[pose["pIC50_range"]]], device=self.device
                ).float()
                uncertainty = torch.tensor(
                    [[pose[f"{self.target_prop}_stderr"]]],
                    device=self.device,
                ).float()

                # Make prediction and calculate loss
                pred, pose_preds = self.model(pose)
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
                    [[pose["pIC50_range"]]], device=self.device
                ).float()
                uncertainty = torch.tensor(
                    [[pose[f"{self.target_prop}_stderr"]]],
                    device=self.device,
                ).float()

                # Make prediction and calculate loss
                pred, pose_preds = self.model(pose)
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
            torch.save(self.model.state_dict(), self.output_dir / f"{epoch_idx}.th")
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
                if isinstance(self.es, BestEarlyStopping) and self.es.check(
                    epoch_idx, epoch_val_loss, self.model.state_dict()
                ):
                    print(
                        (
                            f"Stopping training after epoch {epoch_idx}, "
                            f"using weights from epoch {self.es.best_epoch}"
                        ),
                        flush=True,
                    )
                    self.model.load_state_dict(self.es.best_wts)
                    if self.use_wandb or self.sweep:
                        wandb.log(
                            {
                                "best_epoch": self.es.best_epoch,
                                "best_loss": self.es.best_loss,
                            }
                        )
                    break
                elif isinstance(self.es, ConvergedEarlyStopping) and self.es.check(
                    epoch_val_loss
                ):
                    print(f"Stopping training after epoch {epoch_idx}", flush=True)
                    break

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
