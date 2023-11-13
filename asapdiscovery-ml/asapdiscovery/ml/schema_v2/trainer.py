from typing import Callable

import mtenn
import torch
from asapdiscovery.ml.dataset import DockedDataset, GraphDataset, GroupedDockedDataset
from asapdiscovery.ml.schema_v2.config import ModelConfigBase, OptimizerConfig
from pydantic import BaseModel, Field


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
    dataset: DockedDataset | GroupedDockedDataset | GraphDataset = Field(
        ..., description="Dataset object to train on."
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
    n_epochs: int = Field(300, description="Number of epochs to train for.")
    batch_size: int = Field(
        1, description="Number of samples to predict on before performing backprop."
    )
    target_prop: str = Field("pIC50", description="Target property to train against.")

    # Tracker to make sure the optimizer and ML model are built before trying to train
    _is_initialized = False

    def initialize(self):
        """
        Build the Optimizer and ML Model described by the stored config.
        """
        # Build the Optimizer
        self.optimizer = self.optimizer_config.build()

        # Build the Model
        self.model = self.model_config.build()

        # Set internal tracker to True so we know we can start training
        self._is_initialized = True

    def train(self) -> (mtenn.Model, dict):
        """
        Train the model, returning the trained model and loss_dict.

        Returns
        -------
        mtenn.Model
            Trained model
        dict
            Loss dict
        """
        if not self._is_initialized:
            if self.auto_init:
                self.initialize()
            else:
                raise ValueError("Trainer was not initialized before trying to train.")
