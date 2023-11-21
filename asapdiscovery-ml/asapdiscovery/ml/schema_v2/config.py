from __future__ import annotations

import abc
import pickle as pkl
from collections.abc import Iterator
from pathlib import Path
from typing import Callable, ClassVar

import mtenn
import numpy as np
import torch
from asapdiscovery.data.enum import StringEnum
from asapdiscovery.data.schema import ExperimentalCompoundDataUpdate
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.ml.dataset import DockedDataset, GraphDataset, GroupedDockedDataset
from asapdiscovery.ml.es import BestEarlyStopping, ConvergedEarlyStopping
from pydantic import BaseModel, Field, root_validator


class OptimizerType(StringEnum):
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
            "Type of optimizer to use. "
            f"Options are [{', '.join(OptimizerType.get_values())}]."
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


class EarlyStoppingType(StringEnum):
    """
    Enum for early stopping classes.
    """

    best = "best"
    converged = "converged"


class EarlyStoppingConfig(BaseModel):
    """
    Class for constructing an early stopping class.
    """

    es_type: EarlyStoppingType = Field(
        ...,
        description=(
            "Type of early stopping to use. "
            f"Options are [{', '.join(EarlyStoppingType.get_values())}]."
        ),
    )
    # Parameters for best
    patience: int = Field(
        None,
        description=(
            "The maximum number of epochs to continue training with no improvement in "
            "the val loss. Used only in BestEarlyStopping."
        ),
    )
    # Parameters for converged
    n_check: int = Field(
        None,
        description=(
            "Number of past epochs to keep track of when calculating divergence. "
            "Used only in ConvergedEarlyStopping."
        ),
    )
    divergence: float = Field(
        None,
        description=(
            "Max allowable difference from the mean of the losses. "
            "Used only in ConvergedEarlyStopping."
        ),
    )

    @root_validator(pre=False)
    def check_args(cls, values):
        match values["es_type"]:
            case EarlyStoppingType.best:
                assert (
                    values["patience"] is not None
                ), "Value required for patience when using BestEarlyStopping."
            case EarlyStoppingType.converged:
                assert (values["n_check"] is not None) and (
                    values["divergence"] is not None
                ), (
                    "Values required for n_check and divergence when using "
                    "ConvergedEarlyStopping."
                )
            case other:
                raise ValueError(f"Unknown EarlyStoppingType: {other}")

    def build(self) -> BestEarlyStopping | ConvergedEarlyStopping:
        match self.es_type:
            case EarlyStoppingType.best:
                return BestEarlyStopping(self.patience)
            case EarlyStoppingType.converged:
                return ConvergedEarlyStopping(self.n_check, self.divergence)
            case other:
                raise ValueError(f"Unknown EarlyStoppingType: {other}")


class DatasetType(StringEnum):
    """
    Enum for different Dataset types.
    """

    graph = "graph"
    structural = "structural"


class DatasetConfig(BaseModel):
    """
    Class for constructing an ML Dataset class.
    """

    # Graph or structure-based dataset
    ds_type: DatasetType = Field(
        ...,
        description=(
            "Type of dataset to build. "
            f"Options are [{', '.join(DatasetType.get_values())}]."
        ),
    )

    # Required inputs used to build the dataset
    exp_data: ExperimentalCompoundDataUpdate = Field(
        ..., description="Experimental data."
    )
    input_data: list[Complex] | list[Ligand] = Field(
        ...,
        description=(
            "List of Complex objects (structure-based models) or Ligand objects "
            "(graph-based models), which will be used to generate the input structures."
        ),
    )

    # Cache file to save to/load from. Probably want to move away from pickle eventually
    cache_file: Path | None = Field(
        None, description="Pickle cache file of the actual dataset object."
    )

    # Parallelize data processing
    num_workers: int = Field(
        1, description="Number of threads to use for dataset processing."
    )

    # Multi-pose or not
    grouped: bool = Field(False, description="Build a GroupedDockedDataset.")

    @root_validator(pre=False)
    def check_data_type(cls, values):
        inp = values["input_data"][0]
        match values["ds_type"]:
            case DatasetType.graph:
                assert isinstance(inp, Ligand), (
                    "Expected Ligand input data for graph-based model, but got "
                    f"{type(inp)}."
                )
            case DatasetType.structural:
                assert isinstance(inp, Complex), (
                    "Expected Complex input data for structure-based model, but got "
                    f"{type(inp)}."
                )
            case other:
                raise ValueError(f"Unknown dataset type {other}.")

        return values

    def build(self):
        # Load from the cache file if it exists
        if self.cache_file and self.cache_file.exists():
            return pkl.loads(self.cache_file.read_bytes())

        # Convert ExperimentalCompoundDataUpdate to a dict of relevant info
        exp_dict = self.exp_dict()

        # Build directly from Complexes/Ligands
        #  (still needs to be implemented on the Dataset side)
        match self.ds_type:
            case DatasetType.graph:
                ds = GraphDataset.from_ligands(self.input_data, exp_dict=exp_dict)
            case DatasetType.structural:
                if self.grouped:
                    ds = GroupedDockedDataset.from_complexes(
                        self.input_data, exp_dict=exp_dict
                    )
                else:
                    ds = DockedDataset.from_complexes(
                        self.input_data, exp_dict=exp_dict
                    )
            case other:
                raise ValueError(f"Unknwon dataset type {other}.")

        if self.cache_file:
            self.cache_file.write_bytes(pkl.dumps(ds))

        return ds


class DatasetSplitterType(StringEnum):
    """
    Enum for different methods of splitting a dataset.
    """

    random = "random"
    temporal = "temporal"


class DatasetSplitterConfig(BaseModel):
    """
    Class for splitting an ML Dataset class.
    """

    # Parameter for splitting
    split_type: DatasetSplitterType = Field(
        ...,
        description=(
            "Method to use for splitting. "
            f"Options are [{', '.join(DatasetSplitterType.get_values())}]."
        ),
    )

    # Multi-pose or not
    grouped: bool = Field(False, description="Build a GroupedDockedDataset.")

    # Split sizes
    train_frac: float = Field(
        0.8, description="Fraction of dataset to put in the train split."
    )
    val_frac: float = Field(
        0.1, description="Fraction of dataset to put in the val split."
    )
    test_frac: float = Field(
        0.1, description="Fraction of dataset to put in the test split."
    )
    enforce_1: bool = Field(
        True, description="Make sure that all split fractions add up to 1."
    )

    # Seed for randomly splitting data
    rand_seed: int | None = Field(
        None, description="Random seed to use if randomly splitting data."
    )

    @root_validator(pre=False)
    def check_frac_sum(cls, values):
        frac_sum = sum([values["train_frac"], values["val_frac"], values["test_frac"]])
        if frac_sum < 0:
            raise ValueError("Can't have negative split fractions.")

        if not np.isclose(frac_sum, 1):
            warn_msg = f"Split fractions add to {frac_sum:0.2f}, not 1."
            if values["enforce_1"]:
                raise ValueError(warn_msg)
            else:
                from warnings import warn

                warn(warn_msg, RuntimeWarning)

            if frac_sum > 1:
                raise ValueError("Can't have split fractions adding to > 1.")

        return values

    def split(self, ds: DockedDataset | GraphDataset | GroupedDockedDataset):
        """
        Dispatch method for spliting ds into train, val, and test splits.

        Parameters
        ----------
        ds : DockedDataset | GraphDataset | GroupedDockedDataset
            Full ML dataset to split

        Returns
        -------
        torch.nn.Subset
            Train split
        torch.nn.Subset
            Val split
        torch.nn.Subset
            Test split
        """
        match self.split_type:
            case DatasetSplitterType.random:
                return self._split_random(ds)
            case DatasetSplitterType.temporal:
                return self._split_temporal(ds)
            case other:
                raise ValueError(f"Unknown DatasetSplitterType {other}.")

    @staticmethod
    def _make_subsets(ds, idx_lists, split_lens):
        """
        Helper script for making subsets of a dataset.

        Parameters
        ----------
        ds : Union[cml.data.DockedDataset, cml.data.GraphDataset]
            Molecular dataset to split
        idx_dict : List[List[int]]
            List of lists of indices into `ds`
        split_lens : List[int]
            List of split lengths

        Returns
        -------
        List[torch.utils.data.Subset]
            List of Subsets of original dataset
        """
        # For each Subset, grab all molecules with the included compound_ids
        all_subsets = []
        # Keep track of which structure indices we've seen so we don't double count in the
        #  end splits
        seen_idx = set()
        prev_idx = 0
        # Go up to the last split so we can add anything that got left out from rounding
        for i, n_mols in enumerate(split_lens[:-1]):
            n_mols_cur = 0
            subset_idx = []
            cur_idx = prev_idx
            # Keep adding groups until the split is as big as it needs to be, or we reach
            #  the end of the array (making sure to save at least some molecules for the
            #  rest of the splits)
            while (n_mols_cur < n_mols) and (
                cur_idx < (len(idx_lists) - (len(split_lens) - i))
            ):
                subset_idx.extend(idx_lists[cur_idx])
                n_mols_cur += len(idx_lists[cur_idx])
                cur_idx += 1

            # Make sure we're not including something that's in another split
            subset_idx = [i for i in subset_idx if i not in seen_idx]
            seen_idx.update(subset_idx)
            all_subsets.append(torch.utils.data.Subset(ds, subset_idx))

            # Update counter
            prev_idx = cur_idx

        # Finish up anything leftover
        subset_idx = [i for d in idx_lists[prev_idx:] for i in d if i not in seen_idx]
        all_subsets.append(torch.utils.data.Subset(ds, subset_idx))

        return all_subsets

    def _split_random(self, ds: DockedDataset | GraphDataset | GroupedDockedDataset):
        """
        Random split.

        Parameters
        ----------
        ds : DockedDataset | GraphDataset | GroupedDockedDataset
            Full ML dataset to split

        Returns
        -------
        torch.nn.Subset
            Train split
        torch.nn.Subset
            Val split
        torch.nn.Subset
            Test split
        """

        if self.rand_seed is not None:
            g = torch.Generator()
        else:
            g = torch.Generator().manual_seed(self.rand_seed)
        print("splitting with random seed:", g.initial_seed(), flush=True)

        if self.grouped:
            # Grouped models are already grouped by compound, so don't need to do
            #  anything fancy here
            ds_train, ds_val, ds_test = torch.utils.data.random_split(
                ds, [self.train_frac, self.val_frac, self.test_frac], g
            )
        else:
            # Calculate how many molecules we want covered through each split
            n_mols_split = np.floor(
                np.asarray([self.train_frac, self.val_frac, self.test_frac]) * len(ds)
            )

            # First get all the unique compound_ids
            compound_ids_dict = {}
            for c, idx_list in ds.compounds.items():
                try:
                    compound_ids_dict[c[1]].extend(idx_list)
                except KeyError:
                    compound_ids_dict[c[1]] = idx_list
            all_compound_ids = np.asarray(list(compound_ids_dict.keys()))

            # Shuffle the indices
            indices = torch.randperm(len(all_compound_ids), generator=g)
            idx_lists = [compound_ids_dict[all_compound_ids[i]] for i in indices]

            # For each Subset, grab all molecules with the included compound_ids
            ds_train, ds_val, ds_test = DatasetSplitterConfig._make_subsets(
                ds, idx_lists, n_mols_split
            )

        return ds_train, ds_val, ds_test

    def _split_temporal(self, ds: DockedDataset | GraphDataset | GroupedDockedDataset):
        """
        Temporal split.

        Parameters
        ----------
        ds : DockedDataset | GraphDataset | GroupedDockedDataset
            Full ML dataset to split

        Returns
        -------
        torch.nn.Subset
            Train split
        torch.nn.Subset
            Val split
        torch.nn.Subset
            Test split
        """
        split_fracs = [self.train_frac, self.val_frac, self.test_frac]
        # Check that split_fracs adds to 1, padding if it doesn't
        # Add an allowance for floating point inaccuracies
        total_splits = sum(split_fracs)
        if not np.isclose(total_splits, 1):
            sink_frac = 1 - total_splits
            split_fracs = split_fracs[:2] + [sink_frac] + split_fracs[2:]
            sink_split = True
        else:
            sink_split = False

        # Calculate how many molecules we want covered through each split
        n_mols_split = np.floor(np.asarray(split_fracs) * len(ds))

        # First get all the unique created dates
        dates_dict = {}
        # If we have a grouped dataset, we want to iterate through compound_ids, which will
        #  allow us to access a group of structures. Otherwise, loop through the structures
        #  directly
        if self.grouped:
            iter_list = ds.compound_ids
        else:
            iter_list = ds.structures
        for i, iter_item in enumerate(iter_list):
            if self.grouped:
                # Take the earliest date from all structures (they should all be the same,
                #  but just in case)
                all_dates = [
                    s["date_created"]
                    for s in ds.structures[iter_item]
                    if "date_created" in s
                ]
                if len(all_dates) == 0:
                    raise ValueError("Dataset doesn't contain dates.")
                else:
                    date_created = min(all_dates)
            else:
                try:
                    date_created = iter_item["date_created"]
                except KeyError:
                    raise ValueError("Dataset doesn't contain dates.")
            try:
                dates_dict[date_created].append(i)
            except KeyError:
                dates_dict[date_created] = [i]
        all_dates = np.asarray(list(dates_dict.keys()))

        # Sort the dates
        all_dates_sorted = sorted(all_dates)

        # Make subsets
        idx_lists = [dates_dict[d] for d in all_dates_sorted]
        all_subsets = DatasetSplitterConfig._make_subsets(ds, idx_lists, n_mols_split)

        # Take out the sink split
        if sink_split:
            all_subsets = all_subsets[:2] + all_subsets[3]

        return all_subsets


class ModelType(StringEnum):
    """
    Enum for model types.
    """

    gat = "gat"
    schnet = "schnet"
    e3nn = "e3nn"
    INVALID = "INVALID"


class MTENNStrategy(StringEnum):
    """
    Enum for possible MTENN Strategy classes.
    """

    # delta G strategy
    delta = "delta"
    # ML concatenation strategy
    concat = "concat"
    # Complex-only strategy
    complex = "complex"


class MTENNReadout(StringEnum):
    """
    Enum for possible MTENN Readout classes.
    """

    pic50 = "pic50"


class MTENNCombination(StringEnum):
    """
    Enum for possible MTENN Readout classes.
    """

    mean = "mean"
    max = "max"
    boltzmann = "boltzmann"


class ModelConfigBase(BaseModel):
    model_type: ClassVar[ModelType.INVALID] = ModelType.INVALID

    # Shared parameters for MTENN
    grouped: bool = Field(False, description="Model is a grouped (multi-pose) model.")
    strategy: MTENNStrategy = Field(
        MTENNStrategy.delta,
        description=(
            "Which Strategy to use for combining complex, protein, and ligand "
            "representations in the MTENN Model. "
            f"Options are [{', '.join(MTENNStrategy.get_values())}]."
        ),
    )
    pred_readout: MTENNReadout | None = Field(
        None,
        description=(
            "Which Readout to use for the model predictions. This corresponds "
            "to the individual pose predictions in the case of a GroupedModel. "
            f"Options are [{', '.join(MTENNReadout.get_values())}]."
        ),
    )
    combination: MTENNCombination | None = Field(
        None,
        description=(
            "Which Combination to use for combining predictions in a GroupedModel. "
            f"Options are [{', '.join(MTENNCombination.get_values())}]."
        ),
    )
    comb_readout: MTENNReadout | None = Field(
        None,
        description=(
            "Which Readout to use for the combined model predictions. This is only "
            "relevant in the case of a GroupedModel. "
            f"Options are [{', '.join(MTENNReadout.get_values())}]."
        ),
    )

    # Parameters for MaxCombination
    max_comb_neg: bool = Field(
        True,
        description=(
            "Whether to take the min instead of max when combining pose predictions "
            "with MaxCombination."
        ),
    )
    max_comb_scale: float = Field(
        1000,
        description=(
            "Scaling factor for values when taking the max/min when combining pose "
            "predictions with MaxCombination. A value of 1 will approximate the "
            "Boltzmann mean, while a larger value will more accurately approximate the "
            "max/min operation."
        ),
    )

    # Parameters for PIC50Readout for pred_readout
    pred_substrate: float | None = Field(
        None,
        description=(
            "Substrate concentration to use when using the Cheng-Prusoff equation to "
            "convert deltaG -> IC50 in PIC50Readout for pred_readout. Assumed to be in "
            "the same units as pred_km."
        ),
    )
    pred_km: float | None = Field(
        None,
        description=(
            "Km value to use when using the Cheng-Prusoff equation to convert "
            "deltaG -> IC50 in PIC50Readout for pred_readout. Assumed to be in "
            "the same units as pred_substrate."
        ),
    )

    # Parameters for PIC50Readout for comb_readout
    comb_substrate: float | None = Field(
        None,
        description=(
            "Substrate concentration to use when using the Cheng-Prusoff equation to "
            "convert deltaG -> IC50 in PIC50Readout for comb_readout. Assumed to be in "
            "the same units as comb_km."
        ),
    )
    comb_km: float | None = Field(
        None,
        description=(
            "Km value to use when using the Cheng-Prusoff equation to convert "
            "deltaG -> IC50 in PIC50Readout for comb_readout. Assumed to be in "
            "the same units as comb_substrate."
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
                mtenn_combination = mtenn.combination.MaxCombination(
                    neg=self.max_comb_neg, scale=self.max_comb_scale
                )
            case MTENNCombination.boltzmann:
                mtenn_combination = mtenn.combination.BoltzmannCombination()
            case None:
                mtenn_combination = None

        match self.pred_readout:
            case MTENNReadout.pic50:
                mtenn_pred_readout = mtenn.readout.PIC50Readout(
                    substrate=self.pred_substrate, Km=self.pred_km
                )
            case None:
                mtenn_pred_readout = None

        match self.comb_readout:
            case MTENNReadout.pic50:
                mtenn_comb_readout = mtenn.readout.PIC50Readout(
                    substrate=self.comb_substrate, Km=self.comb_km
                )
            case None:
                mtenn_comb_readout = None

        mtenn_params = {
            "combination": mtenn_combination,
            "pred_readout": mtenn_pred_readout,
            "comb_readout": mtenn_comb_readout,
        }

        # Build the actual Model
        return self._build(mtenn_params)

    def update(self, config_updates={}) -> ModelConfigBase:
        return self._update(config_updates)

    def _update(self, config_updates={}) -> ModelConfigBase:
        """
        Default version of this function. Just update original config with new options,
        and generate new object. Designed to be overloaded if there are specific things
        that a class needs to handle (see GATModelConfig as an example).
        """

        orig_config = self.dict()

        # Get new config by overwriting old stuff with any new stuff
        new_config = orig_config | config_updates

        return type(self)(**new_config)

    @staticmethod
    def _check_grouped(values):
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
    * If a list of values is passed for any parameters, all parameters must be lists of
    the same size, or single values. For parameters that are single values, that same
    value will be used for each layer. For parameters that are lists, those lists will
    be used

    Parameters passed as strings are assumed to be comma-separated lists, and will first
    be cast to lists of the appropriate type, and then processed as described above.

    If lists of multiple different (non-1) sizes are found, an error will be raised.

    Default values here are the default values given in DGL-LifeSci.
    """

    from dgllife.utils import CanonicalAtomFeaturizer

    LIST_PARAMS: ClassVar[dict] = {
        "hidden_feats": int,
        "num_heads": int,
        "feat_drops": float,
        "attn_drops": float,
        "alphas": float,
        "residuals": bool,
        "agg_modes": str,
        "activations": None,
        "biases": bool,
    }

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
    hidden_feats: str | list[int] = Field(
        32,
        description=(
            "Output size of each GAT layer. If an int is passed, the value for "
            "num_layers will be used to determine the size of the model. If a list of "
            "ints is passed, the size of the model will be inferred from the length of "
            "the list."
        ),
    )
    num_heads: str | list[int] = Field(
        4,
        description=(
            "Number of attention heads for each GAT layer. Passing an int or list of "
            "ints functions similarly as for hidden_feats."
        ),
    )
    feat_drops: str | list[float] = Field(
        0,
        description=(
            "Dropout of input features for each GAT layer. Passing an float or list of "
            "floats functions similarly as for hidden_feats."
        ),
    )
    attn_drops: str | list[float] = Field(
        0,
        description=(
            "Dropout of attention values for each GAT layer. Passing an float or list "
            "of floats functions similarly as for hidden_feats."
        ),
    )
    alphas: str | list[float] = Field(
        0.2,
        description=(
            "Hyperparameter for LeakyReLU gate for each GAT layer. Passing an float or "
            "list of floats functions similarly as for hidden_feats."
        ),
    )
    residuals: str | list[bool] = Field(
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
    activations: list[Callable] | None = Field(
        None,
        description=(
            "Activation function for each GAT layer. Passing a function or "
            "list of functions functions similarly as for hidden_feats."
        ),
    )
    biases: str | list[bool] = Field(
        True,
        description=(
            "Whether to use bias for each GAT layer. Passing a bool or "
            "list of bools functions similarly as for hidden_feats."
        ),
    )
    allow_zero_in_degree: bool = Field(
        False, description="Allow zero in degree nodes for all graph layers."
    )

    # Internal tracker for if the parameters were originally built from lists or using
    #  num_layers
    _from_num_layers = False

    @root_validator(pre=False)
    def massage_into_lists(cls, values) -> GATModelConfig:
        # First convert string lists to actual lists
        for param, param_type in cls.LIST_PARAMS.items():
            param_val = values[param]
            if isinstance(param_val, str):
                try:
                    param_val = list(map(param_type, param_val.split(",")))
                except ValueError:
                    raise ValueError(
                        f"Unable to parse value {param_val} for parameter {param}. "
                        f"Expected type of {param_type}."
                    )
                values[param] = param_val

        # Get sizes of all lists
        list_lens = {}
        for p in cls.LIST_PARAMS:
            param_val = values[p]
            if not isinstance(param_val, list):
                # Shouldn't be possible at this point but just in case
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
        elif list_lens_set == {1}:
            # If all lists have only one value, we defer to the value passed to
            #  num_layers, as described in the class docstring
            num_layers = values["num_layers"]
            values["_from_num_layers"] = True
        else:
            num_layers = max(list_lens_set)
            values["_from_num_layers"] = False

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

        pred_readout = mtenn_params.get("pred_readout", None)
        return GAT.get_model(model=model, pred_readout=pred_readout, fix_device=True)

    def _update(self, config_updates={}) -> GATModelConfig:
        orig_config = self.dict()
        if self._from_num_layers:
            # If originally generated from num_layers, want to pull out the first entry
            #  in each list param so it can be re-broadcast with (potentially) new
            #  num_layers
            for param_name in GATModelConfig.LIST_PARAMS.keys():
                orig_config[param_name] = orig_config[param_name][0]

        # Get new config by overwriting old stuff with any new stuff
        new_config = orig_config | config_updates

        # A bit hacky, maybe try and change?
        if isinstance(new_config["activations"], list) and (
            new_config["activations"][0] is None
        ):
            new_config["activations"] = None

        return GATModelConfig(**new_config)


class SchNetModelConfig(ModelConfigBase):
    """
    Class for constructing a SchNet ML model. Default values here are the default values
    given in PyG.
    """

    model_type: ClassVar[ModelType.schnet] = ModelType.schnet

    hidden_channels: int = Field(128, description="Hidden embedding size.")
    num_filters: int = Field(
        128, description="Number of filters to use in the cfconv layers."
    )
    num_interactions: int = Field(6, description="Number of interaction blocks.")
    num_gaussians: int = Field(
        50, description="Number of gaussians to use in the interaction blocks."
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
    atomref: list[float] | None = Field(
        None,
        description=(
            "Reference values for single-atom properties. Should have length of 100 to "
            "match with PyG."
        ),
    )

    @root_validator(pre=False)
    def validate(cls, values):
        # Make sure the grouped stuff is properly assigned
        ModelConfigBase._check_grouped(values)

        # Make sure atomref length is correct (this is required by PyG)
        atomref = values["atomref"]
        if (atomref is not None) and (len(atomref) != 100):
            raise ValueError(f"atomref must be length 100 (got {len(atomref)})")

        return values

    def _build(self, mtenn_params={}):
        """
        Build an MTENN SchNet Model from this config.

        Parameters
        ----------
        mtenn_params: dict
            Dict giving the MTENN Readout. This will be passed by the `build` method in
            the abstract base class

        Returns
        -------
        mtenn.model.Model
            MTENN SchNet Model/GroupedModel
        """
        from mtenn.conversion_utils import SchNet

        # Create an MTENN SchNet model from PyG SchNet model
        model = SchNet(
            hidden_channels=self.hidden_channels,
            num_filters=self.num_filters,
            num_interactions=self.num_interactions,
            num_gaussians=self.num_gaussians,
            interaction_graph=self.interaction_graph,
            cutoff=self.cutoff,
            max_num_neighbors=self.max_num_neighbors,
            readout=self.readout,
            dipole=self.dipole,
            mean=self.mean,
            std=self.std,
            atomref=self.atomref,
        )

        combination = mtenn_params.get("combination", None)
        pred_readout = mtenn_params.get("pred_readout", None)
        comb_readout = mtenn_params.get("comb_readout", None)

        return SchNet.get_model(
            model=model,
            grouped=self.grouped,
            fix_device=True,
            strategy=self.strategy,
            combination=combination,
            pred_readout=pred_readout,
            comb_readout=comb_readout,
        )


class E3NNModelConfig(ModelConfigBase):
    """
    Class for constructing an e3nn ML model.
    """

    model_type: ClassVar[ModelType.e3nn] = ModelType.e3nn

    num_atom_types: int = Field(
        100,
        description=(
            "Number of different atom types. In general, this will just be the "
            "max atomic number of all input atoms."
        ),
    )
    irreps_hidden: dict[str, int] | str = Field(
        {"0": 10, "1": 3, "2": 2, "3": 1},
        description=(
            "Irreps for the hidden layers of the network. "
            "This can either take the form of an Irreps string, or a dict mapping "
            "L levels (parity optional) to the number of Irreps of that level. "
            "If parity is not passed for a given level, both parities will be used. If "
            "you only want one parity for a given level, make sure you specify it. "
            "A dict can also be specified as a string, in the format of a comma "
            "separated list of <irreps_l>:<num_irreps>."
        ),
    )
    lig: bool = Field(
        False, description="Include ligand labels as a node attribute information."
    )
    irreps_edge_attr: int = Field(
        3,
        description=(
            "Which level of spherical harmonics to use for encoding edge attributes "
            "internally."
        ),
    )
    num_layers: int = Field(3, description="Number of network layers.")
    neighbor_dist: float = Field(
        10, description="Cutoff distance for including atoms as neighbors."
    )
    num_basis: int = Field(
        10, description="Number of bases on which the edge length are projected."
    )
    num_radial_layers: int = Field(1, description="Number of radial layers.")
    num_radial_neurons: int = Field(
        128, description="Number of neurons in each radial layer."
    )
    num_neighbors: float = Field(25, description="Typical number of neighbor nodes.")
    num_nodes: float = Field(4700, description="Typical number of nodes in a graph.")

    @root_validator(pre=False)
    def massage_irreps(cls, values):
        from e3nn import o3

        # First just check that the grouped stuff is properly assigned
        ModelConfigBase._check_grouped(values)

        # Now deal with irreps
        irreps = values["irreps_hidden"]
        # First see if this string should be converted into a dict
        if isinstance(irreps, str):
            if ("," in irreps) and (":" in irreps):
                orig_irreps = irreps
                irreps = [i.split(":") for i in irreps.split(",")]
                try:
                    irreps = {
                        irreps_l: int(num_irreps) for irreps_l, num_irreps in irreps
                    }
                except ValueError:
                    raise ValueError(
                        f"Unable to parse irreps dict string: {orig_irreps}"
                    )
            else:
                # If not, try and convert directly to Irreps
                try:
                    _ = o3.Irreps(irreps)
                except ValueError:
                    raise ValueError(f"Invalid irreps string: {irreps}")

                # If already in a good string, can just return
                return values

        # If we got a dict, need to massage that into an Irreps string
        # First make a copy of the input dict in case of errors
        orig_irreps = irreps.copy()
        # Find L levels that got an unspecified parity
        unspecified_l = [k for k in irreps.keys() if ("o" not in k) and ("e" not in k)]
        for irreps_l in unspecified_l:
            num_irreps = irreps.pop(irreps_l)
            irreps[f"{irreps_l}o"] = num_irreps
            irreps[f"{irreps_l}e"] = num_irreps

        # Combine Irreps into str
        irreps = "+".join(
            [f"{num_irreps}x{irrep}" for irrep, num_irreps in irreps.items()]
        )

        # Make sure this Irreps string is valid
        try:
            _ = o3.Irreps(irreps)
        except ValueError:
            raise ValueError(f"Couldn't parse irreps dict: {orig_irreps}")

        values["irreps_hidden"] = irreps
        return values

    def _build(self, mtenn_params={}):
        from e3nn.o3 import Irreps
        from mtenn.conversion_utils import E3NN

        model = E3NN(
            irreps_in=f"{self.num_atom_types}x0e",
            irreps_hidden=self.irreps_hidden,
            irreps_out="1x0e",
            irreps_node_attr="1x0e" if self.lig else None,
            irreps_edge_attr=Irreps.spherical_harmonics(self.irreps_edge_attr),
            layers=self.num_layers,
            max_radius=self.neighbor_dist,
            number_of_basis=self.num_basis,
            radial_layers=self.num_radial_layers,
            radial_neurons=self.num_radial_neurons,
            num_neighbors=self.num_neighbors,
            num_nodes=self.num_nodes,
            reduce_output=True,
        )

        combination = mtenn_params.get("combination", None)
        pred_readout = mtenn_params.get("pred_readout", None)
        comb_readout = mtenn_params.get("comb_readout", None)

        return E3NN.get_model(
            model=model,
            grouped=self.grouped,
            fix_device=True,
            strategy=self.strategy,
            combination=combination,
            pred_readout=pred_readout,
            comb_readout=comb_readout,
        )
