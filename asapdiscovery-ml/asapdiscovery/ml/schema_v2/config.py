from __future__ import annotations

import abc
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Callable, ClassVar

import mtenn
import torch
from pydantic import BaseModel, Field, root_validator


class OptimizerType(str, Enum):
    """
    Enum for training optimizers.
    """

    sgd = "sgd"
    adam = "adam"
    adadelta = "adadelta"
    adamw = "adamw"


class OptimizerConfig(BaseModel):
    """
    Class for constructing an ML optimizer. All parameter defaults are their defaults in
    pytorch.

    NOTE: some of the parameters have different defaults between different optimizers,
    need to figure out how to deal with that
    """

    optimizer_type: OptimizerType = Field(
        OptimizerType.adam,
        description=(
            "Tyoe of optimizer to use. Options are [sgd, adam, adadelta, adamw]."
        ),
    )
    # Common parameters
    lr: float = Field(0.0001, description="Optimizer learning rate.")
    weight_decay: float = Field(0, description="Optimizer weight decay (L2 penalty).")

    # SGD-only parameters
    momentum: float = Field(0, description="Momentum for SGD optimizer.")
    dampening: float = Field(0, description="Dampening for momentum for SGD optimizer.")

    # Adam* parameters
    b1: float = Field(0.9, description="B1 parameter for Adam and AdamW optimizers.")
    b2: float = Field(0.999, description="B2 parameter for Adam and AdamW optimizers.")
    eps: float = Field(
        1e-8, description="Epsilon parameter for Adam, AdamW, and Adadelta optimizers."
    )

    # Adadelta parameters
    rho: float = Field(0.9, description="Rho parameter for Adadelta optimizer.")

    def build(
        self, parameters: Iterator[torch.nn.parameter.Parameter]
    ) -> torch.optim.Optimizer:
        """
        Build the Optimizer object.

        Parameters
        ----------
        parameters : Iterator[torch.nn.parameter.Parameter]
            Model parameters that will be adjusted by the optimizer

        Returns
        -------
        torch.optim.Optimizer
        Optimizer object
        """
        match self.optimizer_type:
            case OptimizerType.sgd:
                return torch.optim.SGD(
                    parameters,
                    lr=self.lr,
                    momentum=self.momentum,
                    dampening=self.dampening,
                    weight_decay=self.weight_decay,
                )
            case OptimizerType.adam:
                return torch.optim.Adam(
                    parameters,
                    lr=self.lr,
                    betas=(self.b1, self.b2),
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                )
            case OptimizerType.adadelta:
                return torch.optim.Adadelta(
                    parameters,
                    rho=self.rho,
                    eps=self.eps,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
            case OptimizerType.adamw:
                return torch.optim.AdamW(
                    parameters,
                    lr=self.lr,
                    betas=(self.b1, self.b2),
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                )
            case optimizer_type:
                # Shouldn't be possible but just in case
                raise ValueError(f"Unknown value for optimizer_type: {optimizer_type}")


class ModelType(str, Enum):
    """
    Enum for model types.
    """

    gat = "gat"
    schnet = "schnet"
    e3nn = "e3nn"
    INVALID = "INVALID"


class MTENNStrategy(str, Enum):
    """
    Enum for possible MTENN Strategy classes.
    """

    # delta G strategy
    delta = "delta"
    # ML concatenation strategy
    concat = "concat"
    # Complex-only strategy
    complex = "complex"


class MTENNReadout(str, Enum):
    """
    Enum for possible MTENN Readout classes.
    """

    pic50 = "pic50"


class MTENNCombination(str, Enum):
    """
    Enum for possible MTENN Readout classes.
    """

    mean = "mean"
    max = "max"
    boltzmann = "boltzmann"


class ModelConfigBase(BaseModel):
    model_type: ClassVar[ModelType.INVALID] = ModelType.INVALID

    # Not sure if I can do this...
    grouped: bool = Field(False, description="Model is a grouped (multi-pose) model.")
    strategy: MTENNStrategy = Field(
        MTENNStrategy.delta,
        description=(
            "Which Strategy to use for combining complex, protein, and ligand "
            "representations in the MTENN Model."
        ),
    )
    pred_readout: MTENNReadout | None = Field(
        None,
        description=(
            "Which Readout to use for the model predictions. This corresponds "
            "to the individual pose predictions in the case of a GroupedModel."
        ),
    )
    combination: MTENNCombination | None = Field(
        None,
        description=(
            "Which Combination to use for combining predictions in a GroupedModel."
        ),
    )
    comb_readout: MTENNReadout | None = Field(
        None,
        description=(
            "Which Readout to use for the combined model predictions. This is only "
            "relevant in the case of a GroupedModel."
        ),
    )

    @abc.abstractmethod
    def _build(self, mtenn_params={}) -> mtenn.model.Model:
        ...

    def build(self) -> mtenn.model.Model:
        # First handle the MTENN classes
        match self.combination:
            case MTENNCombination.mean:
                mtenn_combination = mtenn.combination.MeanCombination()
            case MTENNCombination.max:
                mtenn_combination = mtenn.combination.MaxCombination()
            case MTENNCombination.boltzmann:
                mtenn_combination = mtenn.combination.BoltzmannCombination()
            case None:
                mtenn_combination = None

        match self.pred_readout:
            case MTENNReadout.pic50:
                mtenn_pred_readout = mtenn.readout.PIC50Readout()
            case None:
                mtenn_pred_readout = None

        match self.comb_readout:
            case MTENNReadout.pic50:
                mtenn_comb_readout = mtenn.readout.PIC50Readout()
            case None:
                mtenn_comb_readout = None

        mtenn_params = {
            "combination": mtenn_combination,
            "pred_readout": mtenn_pred_readout,
            "comb_readout": mtenn_comb_readout,
        }

        # Build the actual Model
        return self._build(mtenn_params)

    @staticmethod
    def _check_grouped(cls, values):
        """
        Makes sure that a Combination method is passed if using a GroupedModel. Only
        needs to be called for structure-based models.
        """
        if values["grouped"] and (not values["combination"]):
            raise ValueError("combination must be specified for a GroupedModel.")


class GATModelConfig(ModelConfigBase):
    """
    Class for constructing a GAT ML model. Note that there are two methods for defining
    the size of the model:
    * If single values are passed for all parameters, the value of `num_layers` will be
    used as the size of the model, and each layer will have the parameters given
    * If a list of values is passed for any parameters, all parameters must either be
    lists of the same size, or single values. For parameters that are single values,
    that same value will be used for each layer. For parameters that are lists, those
    lists will be used

    If there are parameters that have list values but the lists are different sizes, an
    error will be raised.

    Default values here are the default values given in DGL-LifeSci.
    """

    from dgllife.utils import CanonicalAtomFeaturizer

    model_type: ClassVar[ModelType.gat] = ModelType.gat

    in_feats: int = Field(
        CanonicalAtomFeaturizer().feat_size(),
        description=(
            "Input node feature size. Defaults to size of the CanonicalAtomFeaturizer."
        ),
    )
    num_layers: int = Field(
        2,
        description=(
            "Number of GAT layers. Ignored if a list of values is passed for any "
            "other argument."
        ),
    )
    hidden_feats: int | list[int] = Field(
        32,
        description=(
            "Output size of each GAT layer. If an int is passed, the value for "
            "num_layers will be used to determine the size of the model. If a list of "
            "ints is passed, the size of the model will be inferred from the length of "
            "the list."
        ),
    )
    num_heads: int | list[int] = Field(
        4,
        description=(
            "Number of attention heads for each GAT layer. Passing an int or list of "
            "ints functions similarly as for hidden_feats."
        ),
    )
    feat_drops: float | list[float] = Field(
        0,
        description=(
            "Dropout of input features for each GAT layer. Passing an float or list of "
            "floats functions similarly as for hidden_feats."
        ),
    )
    attn_drops: float | list[float] = Field(
        0,
        description=(
            "Dropout of attention values for each GAT layer. Passing an float or list "
            "of floats functions similarly as for hidden_feats."
        ),
    )
    alphas: float | list[float] = Field(
        0.2,
        description=(
            "Hyperparameter for LeakyReLU gate for each GAT layer. Passing an float or "
            "list of floats functions similarly as for hidden_feats."
        ),
    )
    residuals: bool | list[bool] = Field(
        True,
        description=(
            "Whether to use residual connection for each GAT layer. Passing a bool or "
            "list of bools functions similarly as for hidden_feats."
        ),
    )
    agg_modes: str | list[str] = Field(
        "flatten",
        description=(
            "Which aggregation mode [flatten, mean] to use for each GAT layer. "
            "Passing a str or list of strs functions similarly as for hidden_feats."
        ),
    )
    activations: Callable | list[Callable] | None = Field(
        None,
        description=(
            "Activation function for each GAT layer. Passing a function or "
            "list of functions functions similarly as for hidden_feats."
        ),
    )
    biases: bool | list[bool] = Field(
        True,
        description=(
            "Whether to use bias for each GAT layer. Passing a bool or "
            "list of bools functions similarly as for hidden_feats."
        ),
    )
    allow_zero_in_degree: bool = Field(
        False, description="Allow zero in degree nodes for all graph layers."
    )

    @root_validator(pre=False)
    def massage_into_lists(cls, values) -> GATModelConfig:
        list_params = [
            "hidden_feats",
            "num_heads",
            "feat_drops",
            "attn_drops",
            "alphas",
            "residuals",
            "agg_modes",
            "activations",
            "biases",
        ]
        # First check if any of the list-optional params are lists
        if any([isinstance(values[p], list) for p in list_params]):
            use_num_layers = False
        else:
            use_num_layers = True

        # If all values are just ints/floats/bools (ie no lists), we can just make the
        #  lists based on num_layers and return
        if use_num_layers:
            for p in list_params:
                values[p] = [values[p]] * values["num_layers"]

            return values

        # Otherwise need to do a bit more logic to get things right
        list_lens = {}
        for p in list_params:
            param_val = values[p]
            if not isinstance(param_val, list):
                param_val = [param_val]
                values[p] = param_val
            list_lens[p] = len(param_val)

        # Check that there's only one length present
        list_lens_set = set(list_lens.values())
        # This could be 0 if lists of length 1 were passed, which is valid
        if len(list_lens_set - {1}) > 1:
            raise ValueError(
                "All passed parameter lists must be the same value. "
                f"Instead got list lengths of: {list_lens}"
            )

        num_layers = max(list_lens_set)
        values["num_layers"] = num_layers
        # If we just want a model with one layer, can return early since we've already
        #  converted everything into lists
        if num_layers == 1:
            return values

        # Adjust any length 1 list to be the right length
        for p, list_len in list_lens.items():
            if list_len == 1:
                values[p] = values[p] * num_layers

        return values

    def _build(self, mtenn_params={}):
        """
        Build an MTENN GAT Model from this config.

        Parameters
        ----------
        mtenn_params: dict
            Dict giving the MTENN Readout. This will be passed by the `build` method in
            the abstract base class

        Returns
        -------
        mtenn.model.Model
            MTENN GAT LigandOnlyModel
        """
        from mtenn.conversion_utils import GAT

        model = GAT(
            in_feats=self.in_feats,
            hidden_feats=self.hidden_feats,
            num_heads=self.num_heads,
            feat_drops=self.feat_drops,
            attn_drops=self.attn_drops,
            alphas=self.alphas,
            residuals=self.residuals,
            agg_modes=self.agg_modes,
            activations=self.activations,
            biases=self.biases,
            allow_zero_in_degree=self.allow_zero_in_degree,
        )

        pred_readout = (
            mtenn_params["pred_readout"] if "pred_readout" in mtenn_params else None
        )
        return GAT.get_model(model=model, pred_readout=pred_readout, fix_device=True)


class SchNetModelConfig(ModelConfigBase):
    """
    Class for constructing a GAT ML model. Default values here are the default values
    given in PyG.
    """

    model_type: ClassVar[ModelType.schnet] = ModelType.schnet

    hidden_channels: int = Field(128, description=("Hidden embedding size."))
    num_filters: int = Field(
        128, description=("Number of filters to use in the cfconv layers.")
    )
    num_interactions: int = Field(6, description=("Number of interaction blocks."))
    num_gaussians: int = Field(
        50, description=("Number of gaussians to use in the interaction blocks.")
    )
    interaction_graph: Callable | None = Field(
        None,
        description=(
            "Function to compute the pairwise interaction graph and "
            "interatomic distances."
        ),
    )
    cutoff: float = Field(
        10, description="Cutoff distance for interatomic interactions."
    )
    max_num_neighbors: int = Field(
        32, description="Maximum number of neighbors to collect for each node."
    )
    readout: str = Field(
        "add", description="Which global aggregation to use [add, mean]."
    )
    dipole: bool = Field(
        False,
        description=(
            "Whether to use the magnitude of the dipole moment to make the "
            "final prediction."
        ),
    )
    mean: float | None = Field(
        None,
        description=(
            "Mean of property to predict, to be added to the model prediction before "
            "returning. This value is only used if dipole is False and a value is also "
            "passed for std."
        ),
    )
    std: float | None = Field(
        None,
        description=(
            "Standard deviation of property to predict, used to scale the model "
            "prediction before returning. This value is only used if dipole is False "
            "and a value is also passed for mean."
        ),
    )
    atomref: torch.Tensor | None = Field(
        None,
        description=(
            "Reference values for single-atom properties. Should have shape "
            "(max_atomic_number, )."
        ),
    )
