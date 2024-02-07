import json
import pickle as pkl
from collections.abc import Iterator
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
from asapdiscovery.data.util.stringenum import StringEnum
from asapdiscovery.data.schema.experimental import ExperimentalCompoundData
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.util.utils import extract_compounds_from_filenames
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

        return values

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
    exp_data: dict[str, dict[str, Any]] = Field(
        {},
        description=(
            "Dict mapping from compound_id to another dict containing "
            "experimental data."
        ),
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

    # Dataset will be used in an e3nn model, so make sure the fields have the correct
    #  names
    for_e3nn: bool = Field(False, description="Dataset will be used in an e3nn model.")

    # Don't use (and overwrite) any existing cache_file
    overwrite: bool = Field(False, description="Overwrite any existing cache_file.")

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

    @classmethod
    def from_exp_file(cls, exp_file: Path, **config_kwargs):
        """
        Build a graph DatasetConfig from an experimental data file.

        Parameters
        ----------
        exp_file : Path
            JSON file giving a list of ExperimentalDataCompound objects
        config_kwargs
            Other kwargs that are passed to the class constructor


        Returns
        -------
        DatasetConfig
        """

        # Parse experimental data
        exp_compounds = [
            ExperimentalCompoundData(**d) for d in json.loads(exp_file.read_text())
        ]
        exp_data = {
            c.compound_id: c.experimental_data | {"date_created": c.date_created}
            for c in exp_compounds
        }

        # Update parsed exp_data with anything passed explicitly
        if "exp_data" in config_kwargs:
            exp_data |= config_kwargs["exp_data"]

        # Create Ligand objects from SMILES
        input_data = [
            Ligand.from_smiles(c.smiles, compound_name=c.compound_id)
            for c in exp_compounds
        ]

        # Get rid of the kwargs that we are passing explicitly
        config_kwargs = {
            k: v
            for k, v in config_kwargs.items()
            if k not in {"ds_type", "exp_data", "input_data"}
        }

        return cls(
            ds_type=DatasetType.graph,
            exp_data=exp_data,
            input_data=input_data,
            **config_kwargs,
        )

    @classmethod
    def from_str_files(
        cls,
        structures: str,
        xtal_regex: str,
        cpd_regex: str,
        for_training: bool = False,
        exp_file: Path | None = None,
        **config_kwargs,
    ):
        """
        Build a structural DatasetConfig from structure files.

        Parameters
        ----------
        structures : str
            Glob or directory containing PDB files
        xtal_regex : str
            Regex for extracting crystal structure name from filename
        cpd_regex : str
            Regex for extracting compound id from filename
        for_training : bool, default=False
            This dataset will be used for training. Forces exp_file to exist, and
            filters structures to only include those with exp data
        exp_file : str, optional
            JSON file giving a list of ExperimentalDataCompound objects
        config_kwargs
            Other kwargs that are passed to the class constructor


        Returns
        -------
        DatasetConfig
        """

        # Parse experimental data
        if exp_file and exp_file.exists():
            exp_compounds = [
                ExperimentalCompoundData(**d) for d in json.loads(exp_file.read_text())
            ]
            exp_data = {
                c.compound_id: c.experimental_data | {"date_created": c.date_created}
                for c in exp_compounds
            }

            # Update parsed exp_data with anything passed explicitly
            if "exp_data" in config_kwargs:
                exp_data |= config_kwargs["exp_data"]
        elif for_training:
            raise ValueError(
                "exp_file must be passed if the dataset will be used for training."
            )
        else:
            exp_data = {}

        # Get structure files
        if Path(structures).is_dir():
            all_str_fns = glob(f"{structures}/*.pdb")
        else:
            all_str_fns = glob(structures)
        compounds = extract_compounds_from_filenames(
            all_str_fns, xtal_pat=xtal_regex, compound_pat=cpd_regex, fail_val="NA"
        )

        # Filter compounds to only include datat that we have experimental data for
        if for_training:
            idx = [c[1] in exp_data for c in compounds]
            print(
                f"Filtering {len(idx) - sum(idx)} structures that we don't have",
                "experimental data for.",
                flush=True,
            )
            compounds = [c for i, c in zip(idx, compounds) if i]
            all_str_fns = [fn for i, fn in zip(idx, all_str_fns) if i]

        # Create Complex objects from PDB files
        print(len(all_str_fns), len(compounds), flush=True)
        input_data = [
            Complex.from_pdb(
                fn,
                target_kwargs={"target_name": cpd[0]},
                ligand_kwargs={"compound_name": cpd[1]},
            )
            for fn, cpd in zip(all_str_fns, compounds)
        ]

        # Get rid of the kwargs that we are passing explicitly
        config_kwargs = {
            k: v
            for k, v in config_kwargs.items()
            if k not in {"ds_type", "exp_data", "input_data"}
        }

        return cls(
            ds_type=DatasetType.structural,
            exp_data=exp_data,
            input_data=input_data,
            **config_kwargs,
        )

    def build(self):
        # Load from the cache file if it exists
        if self.cache_file and self.cache_file.exists() and (not self.overwrite):
            print("loading from cache", flush=True)
            ds = pkl.loads(self.cache_file.read_bytes())
            if self.for_e3nn:
                ds = DatasetConfig.fix_e3nn_labels(ds)
            return ds

        # Build directly from Complexes/Ligands
        match self.ds_type:
            case DatasetType.graph:
                from dgllife.utils import CanonicalAtomFeaturizer

                ds = GraphDataset.from_ligands(
                    self.input_data,
                    exp_dict=self.exp_data,
                    node_featurizer=CanonicalAtomFeaturizer(),
                )
            case DatasetType.structural:
                if self.grouped:
                    ds = GroupedDockedDataset.from_complexes(
                        self.input_data, exp_dict=self.exp_data
                    )
                else:
                    ds = DockedDataset.from_complexes(
                        self.input_data, exp_dict=self.exp_data
                    )
                if self.for_e3nn:
                    ds = DatasetConfig.fix_e3nn_labels(ds)
            case other:
                raise ValueError(f"Unknwon dataset type {other}.")

        if self.cache_file:
            self.cache_file.write_bytes(pkl.dumps(ds))

        return ds

    @staticmethod
    def fix_e3nn_labels(ds):
        for _, pose in ds:
            pose["x"] = torch.nn.functional.one_hot(pose["z"] - 1, 100).float()
            pose["z"] = pose["lig"].reshape((-1, 1)).float()

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
        for i, n_mols in enumerate(split_lens):
            n_mols_cur = 0
            subset_idx = []
            cur_idx = prev_idx
            # Keep adding groups until the split is as big as it needs to be, or we reach
            #  the end of the array (making sure to save at least some molecules for the
            #  rest of the splits)
            while (n_mols_cur < n_mols) and (
                cur_idx < (len(idx_lists) - (len(split_lens) - i - 1))
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

        if self.rand_seed is None:
            g = torch.Generator()
            g.seed()
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
            print("compound_ids_dict", next(iter(compound_ids_dict.items())))
            print(
                "all_compound_ids",
                len(all_compound_ids),
                len(set(all_compound_ids)),
                flush=True,
            )

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


class LossFunctionType(StringEnum):
    """
    Enum for different methods of splitting a dataset.
    """

    # Standard MSE loss
    mse = "mse"
    # Stepped MSE loss (adjusted loss for values outside assay range)
    mse_step = "mse_step"
    # Gaussian NLL loss (ignoring semiquant values)
    gaussian = "gaussian"
    # Gaussian NLL loss (including semiquant values)
    gaussian_sq = "gaussian_sq"


class LossFunctionConfig(BaseModel):
    """
    Class for splitting an ML Dataset class.
    """

    # Parameter for splitting
    loss_type: LossFunctionType = Field(
        ...,
        description=(
            "Loss function to use. "
            f"Options are [{', '.join(LossFunctionType.get_values())}]."
        ),
    )

    # Value to fill in for semiquant uncertainty values in gaussian_sq loss
    semiquant_fill: float = Field(
        None,
        description=(
            "Value to fill in for semiquant uncertainty values in gaussian_sq loss."
        ),
    )

    def build(self):
        from asapdiscovery.ml.loss import GaussianNLLLoss, MSELoss

        match self.loss_type:
            case LossFunctionType.mse:
                return MSELoss()
            case LossFunctionType.mse_step:
                return MSELoss("step")
            case LossFunctionType.gaussian:
                return GaussianNLLLoss(keep_sq=False)
            case LossFunctionType.gaussian_sq:
                return GaussianNLLLoss(keep_sq=True, semiquant_fill=self.semiquant_fill)
            case other:
                raise ValueError(f"Unknown LossFunctionType {other}.")
