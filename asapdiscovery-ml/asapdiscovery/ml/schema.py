import json
import multiprocessing as mp
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import pandas
import torch
from asapdiscovery.ml.config import LossFunctionConfig
from pydantic.v1 import BaseModel, Extra, Field, validator
from scipy.stats import bootstrap, kendalltau, spearmanr


class TrainingPrediction(BaseModel):
    """
    Class to store predictions made during training.
    """

    # Identifiers
    compound_id: str = Field(..., description="Compound ID for the ligand.")
    xtal_id: str = Field(..., description="Crystal ID for the protein structure.")

    # Target info
    target_prop: str = Field(..., description="Target property being predicted.")
    target_val: float | torch.Tensor = Field(
        ..., description="Target value to predict."
    )
    in_range: int = Field(
        None,
        description=(
            "Whether target is below (-1), within (0), or above (1) the assay range. "
            "Not always applicable."
        ),
    )
    uncertainty: float = Field(
        None, description="Uncertainty in experimental measurement."
    )

    # Prediction info
    predictions: list[float] = Field([], description="Model prediction.")
    pose_predictions: list[list[float]] = Field(
        [],
        description="Single-pose model prediction for each pose in input.",
    )

    # Loss info
    loss_config: LossFunctionConfig = Field(
        ..., description="Config describing loss function."
    )
    loss_vals: list[float] = Field([], description="Loss value of model prediction.")
    loss_weight: float = Field(
        1.0, description="Contribution of this loss function to the full loss."
    )

    class Config:
        # Allow things to be added to the object after initialization/validation
        extra = Extra.allow

        # Allow torch types
        arbitrary_types_allowed = True

        # Custom encoder to cast device to str before trying to serialize
        json_encoders = {
            torch.Tensor: lambda t: t.tolist(),
        }

    @validator("target_val", pre=True, always=True)
    def cast_target_val(cls, v):
        if isinstance(v, float):
            return v

        if isinstance(v, torch.Tensor):
            return v.clone().detach()

        return torch.tensor(v)

    def to_empty(self):
        """
        Create a copy with none of the tracked values. Useful checking identity.

        Returns
        -------
        TrainingPrediction
            Copy of self without predictions, pose_predictions, or loss_vals
        """

        # Make a copy
        d = self.dict()

        # Get rid of tracked values
        del d["predictions"]
        del d["pose_predictions"]
        del d["loss_vals"]

        return TrainingPrediction(**d)


class TrainingPredictionTracker(BaseModel):
    """
    Class to organize predictions.
    """

    split_dict: dict[str, list[TrainingPrediction]] = Field(
        None, description="Internal dict storing all TrainingPredictions."
    )

    class Config:
        # Allow things to be added to the object after initialization/validation
        extra = Extra.allow

        # Custom encoder to cast device to str before trying to serialize
        json_encoders = {
            torch.Tensor: lambda t: t.tolist(),
        }

    @validator("split_dict", always=True)
    def init_split_dict(cls, split_dict):
        # If nothing was passed, just init an empty dict
        if not split_dict:
            return {"train": [], "val": [], "test": []}

        # Make sure that the format is correct
        if split_dict.keys() != {"train", "val", "test"}:
            raise ValueError(f"Received unexpected dict keys: {split_dict.keys()}")

        # Make sure that each split has a list
        if not all([isinstance(sp_list, list) for sp_list in split_dict.values()]):
            raise ValueError(
                "All dict values must be lists, got "
                f"{[type(v) for v in split_dict.values()]}"
            )

        # Make sure all the lists have the right types in them
        if not all(
            [
                (len(sp_list) == 0)
                or ({type(v) for v in sp_list} == {TrainingPrediction})
                for sp_list in split_dict.values()
            ]
        ):
            raise ValueError(
                "All dict values must only contain TrainingPredictions, got "
                f"{[{type(v) for v in sp_list} for sp_list in split_dict.values()]}"
            )

        return split_dict

    @classmethod
    def from_loss_dict(cls, loss_dict, loss_config, target_prop="pIC50"):
        """
        Method for building a TrainingPredictionTracker from the old loss_dict method
        of tracking losses and predictions over training.

        Parameters
        ----------
        loss_dict : dict
            Old style of loss_dict
        loss_config : LossFunctionConfig
            LossFunctionConfig to set for each TrainingPrediction. loss_dict format only
            allowed for one type of loss, so we only need to pass one
        target_prop : str, default="pIC50"
            Name of target being predicted. Same deal as for loss_config

        Returns
        -------
        cls
            Parsed TrainingPredictionTracker
        """
        tracker_split_dict = {"train": [], "val": [], "test": []}

        # Loop through
        for sp, split_dict in loss_dict.items():
            for compound_id, cpd_dict in split_dict.items():
                tp = TrainingPrediction(
                    compound_id=compound_id,
                    xtal_id="NA",
                    target_prop=target_prop,
                    target_val=cpd_dict["target"],
                    in_range=cpd_dict["in_range"],
                    uncertainty=cpd_dict["uncertainty"],
                    predictions=cpd_dict["preds"],
                    pose_predictions=cpd_dict["pose_preds"],
                    loss_config=loss_config,
                    loss_vals=cpd_dict["losses"],
                )
                tracker_split_dict[sp].append(tp)

        return cls(split_dict=tracker_split_dict)

    def __len__(self):
        return sum([len(split_list) for split_list in self.split_dict.values()])

    def __iter__(self):
        for sp, split_list in self.split_dict.items():
            for tp in split_list:
                yield sp, tp

    def get_compounds(self):
        return {
            sp: {(tp.xtal_id, tp.compound_id) for tp in split_list}
            for sp, split_list in self.split_dict.items()
        }

    def get_compound_ids(self):
        return {
            sp: {tp.compound_id for tp in split_list}
            for sp, split_list in self.split_dict.items()
        }

    def _find_value_idxs(
        self,
        split=None,
        compound_id=None,
        xtal_id=None,
        target_prop=None,
        loss_config=None,
    ):
        """
        Helper function to find the appropriate indices in each list of self.split_dict.

        Parameters
        ----------
        split : str, optional
            Split to look for values in
        compound_id : str, optional
            Compound ID to match
        xtal_id : str, optional
            Crystal structure to match
        target_prop : str, optional
            Target property to match
        loss_config : LossFunctionConfig, optional
            LossFunctionConfig to match

        Returns
        -------
        dict[str, int]
            Dict mapping split to indices in each split list
        """

        # Match functions
        def compound_id_match(query):
            return (compound_id is None) or (query.compound_id == compound_id)

        def xtal_id_match(query):
            return (xtal_id is None) or (query.xtal_id == xtal_id)

        def target_prop_match(query):
            return (target_prop is None) or (query.target_prop == target_prop)

        def loss_config_match(query):
            return (loss_config is None) or (query.loss_config == loss_config)

        if split:
            return {
                split: [
                    i
                    for i, q in enumerate(self.split_dict[split])
                    if compound_id_match(q)
                    and xtal_id_match(q)
                    and target_prop_match(q)
                    and loss_config_match(q)
                ]
            } | {sp: [] for sp in self.split_dict.keys() if sp != split}
        else:
            return {
                sp: [
                    i
                    for i, q in enumerate(split_list)
                    if compound_id_match(q)
                    and xtal_id_match(q)
                    and target_prop_match(q)
                    and loss_config_match(q)
                ]
                for sp, split_list in self.split_dict.items()
            }

    def get_values(
        self,
        split=None,
        compound_id=None,
        xtal_id=None,
        target_prop=None,
        loss_config=None,
    ):
        """
        Get TrainingPrediction values based on passed filters. The type of the return
        value will depend on the filters passed. If split is not passed, the result will
        be a dict, giving a mapping of split: list[TrainingPrediction]. If a split is
        given, then a list of the TrainingPredictions found in that split will be
        returned.

        Parameters
        ----------
        split : str, optional
            Split to look for values in
        compound_id : str, optional
            Compound ID to match
        xtal_id : str, optional
            Crystal structure to match
        target_prop : str, optional
            Target property to match
        loss_config : LossFunctionConfig, optional
            LossFunctionConfig to match

        Returns
        -------
        dict[str, list[TrainingPrediction]] | list[TrainingPrediction]
            Found values
        """

        # Get indices to return
        return_idxs = self._find_value_idxs(
            split, compound_id, xtal_id, target_prop, loss_config
        )

        # Extract values
        if split:
            return [self.split_dict[split][i] for i in return_idxs[split]]
        else:
            return {
                sp: [split_list[i] for i in return_idxs[sp]]
                for sp, split_list in self.split_dict.items()
            }

    def update_values(
        self,
        prediction,
        pose_predictions,
        loss_val,
        split=None,
        compound_id=None,
        xtal_id=None,
        target_prop=None,
        loss_config=None,
        allow_multiple=False,
        **kwargs,
    ):
        """
        Get TrainingPrediction values based on passed filters. The type of the return
        value will depend on the filters passed. If split is not passed, the result will
        be a dict, giving a mapping of split: list[TrainingPrediction]. If a split is
        given, then a list of the TrainingPredictions found in that split will be
        returned. If no values are returned, a new TrainingPrediction will be created
        using the passed search terms, as well as any additional kwargs passed.

        Parameters
        ----------
        prediction : float
            Model prediction value to add
        pose_predictions : list[float]
            Single-pose model prediction values to add
        loss_val : float
            Loss value to add
        split : str, optional
            Split to look for values in
        compound_id : str, optional
            Compound ID to match
        xtal_id : str, optional
            Crystal structure to match
        target_prop : str, optional
            Target property to match
        loss_config : LossFunctionConfig, optional
            LossFunctionConfig to match
        allow_multiple : bool, default=False
            Allow updating multiple entries at once. This is disabled by default, which
            will raise an error in the case that the passed filter criteria return more
            than one entry.
        """

        # Get indices to return
        return_idxs = self._find_value_idxs(
            split, compound_id, xtal_id, target_prop, loss_config
        )

        num_found = sum([len(idx_list) for idx_list in return_idxs.values()])
        if num_found == 0:
            if split not in self.split_dict.keys():
                raise ValueError(
                    "Can't add new TrainingPrediction without split specified."
                )

            new_pred = TrainingPrediction(
                compound_id=compound_id,
                xtal_id=xtal_id,
                target_prop=target_prop,
                predictions=[prediction],
                pose_predictions=[pose_predictions],
                loss_config=loss_config,
                loss_vals=[loss_val],
                **kwargs,
            )
            self.split_dict[split].append(new_pred)

            return

        # Check that we've only got one, if necessary
        if not allow_multiple:
            if num_found > 1:
                raise ValueError(
                    "Multiple results found for search "
                    ", ".join(
                        map(
                            str, [split, compound_id, xtal_id, target_prop, loss_config]
                        )
                    )
                )

        for sp, split_list in self.split_dict.items():
            for i in return_idxs[sp]:
                split_list[i].predictions.append(prediction)
                split_list[i].pose_predictions.append(pose_predictions)
                split_list[i].loss_vals.append(loss_val)

    def get_losses(self, agg_compounds=False, agg_losses=False):
        """
        Convenience function for extracting the per-epoch loss values across all
        tracked values. The output structure will differ depending on the combination of
        (agg_compounds, agg_losses):

        * (False, False): dict with levels split: compound: loss_config: loss_vals
        * (True, False): dict with levels split: loss_config: loss_vals
        * (False, True): dict with levels split: compound: loss_vals
        * (True, True): dict with levels split: loss_vals

        Parameters
        ----------
        agg_compounds : bool, default=False
            Aggregate (by mean) loss values for all compounds
        agg_losses : bool, default=False
            Aggregate (by weighted mean) all different types of loss values for each
            compound

        Returns
        -------
        dict
            Dict storing loss values, as described in docstring
        """

        # Check that everything has the same number of loss_values
        all_loss_vals_lens = {len(tp.loss_vals) for _, tp in self}
        if len(all_loss_vals_lens) > 1:
            raise ValueError("Mismatched number of loss values")

        # Check that all loss_configs are consistent if not collapsing them
        if agg_compounds and (not agg_losses):
            sp_loss_config_dict = {}
            for sp, split_list in self.split_dict.items():
                cur_loss_configs = {}
                for tp in split_list:
                    try:
                        cur_loss_configs[tp.compound_id].update([tp.loss_config.json()])
                    except KeyError:
                        cur_loss_configs[tp.compound_id] = {tp.loss_config.json()}

                cur_loss_configs = {tuple(s) for s in cur_loss_configs.values()}
                if len(cur_loss_configs) > 1:
                    raise ValueError(f"Mismatched loss_configs in split {sp}")
                elif len(cur_loss_configs) == 0:
                    sp_loss_config_dict[sp] = ()
                else:
                    sp_loss_config_dict[sp] = next(iter(cur_loss_configs))

        full_loss_dict = {}

        # Build full dict, then we can aggregate as desired later
        for sp, split_list in self.split_dict.items():
            for tp in split_list:
                # Indexes into full_loss_dict
                idx = [sp, tp.compound_id, tp.loss_config.json()]

                # Keep going into the dict until we reach the bottom level
                cur_d = full_loss_dict
                for val in idx[:-1]:
                    try:
                        cur_d = cur_d[val]
                    except KeyError:
                        cur_d[val] = {}
                        cur_d = cur_d[val]

                loss_vals = np.asarray(tp.loss_vals)
                if agg_losses:
                    # Taking the weighted mean, so need to multiply by weight
                    loss_vals *= tp.loss_weight

                try:
                    cur_d[idx[-1]].append(loss_vals)
                    print(
                        "Multiple TrainingPrediction values found for "
                        f'compound_id="{tp.compound_id}" and '
                        f'loss_config="{tp.loss_config.json()}"'
                    )
                except KeyError:
                    cur_d[idx[-1]] = [loss_vals]

        agg_loss_dict = {}
        # Aggregate stuff, probably bottom up
        for sp, split_dict in full_loss_dict.items():
            if agg_compounds:
                # If aggregating across compounds, keep everything in a list to flatten
                #  later
                sp_loss_vals = []
            else:
                # Otherwise keep track of stuff per-compound
                sp_loss_vals = {}
            for compound_id, cpd_dict in split_dict.items():
                if agg_losses:
                    # First take the mean in case there were multiple TrainingPrediction
                    #  values with the same loss config, then sum across loss configs
                    #  since we've already taken care of weighting the values
                    lc_loss_vals = np.stack(
                        [
                            np.stack(loss_val_lists, axis=0).mean(axis=0)
                            for loss_val_lists in cpd_dict.values()
                        ],
                        axis=0,
                    ).sum(axis=0)
                else:
                    # Just take mean in case of multiple TrainingPrediction values with
                    #  the same loss config
                    # Return a dict since not aggregating across loss config values
                    lc_loss_vals = {
                        loss_config: np.stack(loss_val_lists, axis=0).mean(axis=0)
                        for loss_config, loss_val_lists in cpd_dict.items()
                    }

                if agg_compounds:
                    sp_loss_vals.append(lc_loss_vals)
                else:
                    sp_loss_vals[compound_id] = lc_loss_vals

            if agg_compounds and (not agg_losses):
                # Need to combine across dicts
                sp_loss_vals = {
                    loss_config: np.stack(
                        [cpd_dict[loss_config] for cpd_dict in sp_loss_vals],
                        axis=0,
                    ).mean(axis=0)
                    for loss_config in sp_loss_config_dict[sp]
                }
            elif agg_compounds:
                # Just take mean across compounds
                sp_loss_vals = np.stack(sp_loss_vals, axis=0).mean(axis=0)

            agg_loss_dict[sp] = sp_loss_vals

        return agg_loss_dict

    def get_predictions(self, agg_compounds="none", in_range_only=False):
        """
        Convenience function for extracting the per-epoch predictions values across all
        tracked values. The output structure will differ depending on the selection for
        agg_compounds:

        * "none": dict with levels split: compound: predictions (n_epochs,)
        * "stack": dict with levels split: predictions (n_compounds, n_epochs)
        * "mean": dict with levels split: predictions (n_epochs,)

        Parameters
        ----------
        agg_compounds : str, default="none"
            How to aggregate the compounds. Options are "none", which does no
            aggregation, "stack", which stacks the prediction values for each compound,
            and "mean", which takes the mean across all compounds
        in_range_only : bool, default=False
            Only take predictions for target values that are within the assay range

        Returns
        -------
        dict
            Dict storing predictions, as described in docstring
        """

        agg_compounds = agg_compounds.lower()
        if agg_compounds not in {"none", "stack", "mean"}:
            raise ValueError(f'Unknown value for agg_compounds: "{agg_compounds}"')

        # Check that everything has the same number of epochs
        all_predictions_lens = {len(tp.predictions) for _, tp in self}
        if len(all_predictions_lens) > 1:
            raise ValueError("Mismatched number of predictions")

        # Check that each compound has the same prediction values across loss_configs
        sp_compound_preds_dict = {}
        for sp, split_list in self.split_dict.items():
            cur_preds = {}
            for tp in split_list:
                # If we only want in-range measurements, skip any that are out of range
                if in_range_only and tp.in_range:
                    continue

                try:
                    cur_preds[tp.compound_id].update([tuple(tp.predictions)])
                except KeyError:
                    cur_preds[tp.compound_id] = {tuple(tp.predictions)}

            for compound_id, pred_list_set in cur_preds.items():
                if len(pred_list_set) > 1:
                    raise ValueError(
                        f"Mismatched predictions for compound {compound_id} in "
                        f"split {sp}"
                    )

            cur_preds = {
                compound_id: next(iter(pred_list_set))
                for compound_id, pred_list_set in cur_preds.items()
            }
            sp_compound_preds_dict[sp] = cur_preds

        # Format into numpy arrays
        sp_compound_preds_dict = {
            sp: {
                compound_id: np.asarray(pred_set).flatten()
                for compound_id, pred_set in split_dict.items()
            }
            for sp, split_dict in sp_compound_preds_dict.items()
        }
        if agg_compounds == "none":
            return sp_compound_preds_dict
        elif agg_compounds == "stack":
            return {
                sp: np.stack(list(split_dict.values()))
                for sp, split_dict in sp_compound_preds_dict.items()
            }
        else:
            return {
                sp: np.stack(list(split_dict.values())).mean(axis=0)
                for sp, split_dict in sp_compound_preds_dict.items()
            }

        return sp_compound_preds_dict

    def get_target_vals(self, target_prop, flatten_compounds=False, return_range=False):
        """
        Convenience function for extracting the per-compound target values across. The
        output structure will differ depending on the selection for flatten_compounds:

        * False: dict with levels split: compound: target val (scalar)
        * True: dict with levels split: target vals (n_compounds,)

        If return_range is True, this function will also return the in-range values for
        each compound. The shape of this output will be the same as the target val
        output.

        Parameters
        ----------
        target_prop : str
            Which target property to pull
        flatten_compounds : bool, default=False
            Whether to combine the target values into a single tensor for each split
            (True) or leave the return value as a nested dict (False, default)
        return_range : bool, default=False
            Return the in-range values as well as the target values

        Returns
        -------
        dict
            Dict storing predictions, as described in docstring
        """
        # First get values for this target
        target_prop_vals = self.get_values(target_prop=target_prop)

        # Loop through and keep track of target val for each compound
        target_val_dict = {sp: {} for sp in target_prop_vals.keys()}
        in_range_dict = {sp: {} for sp in target_prop_vals.keys()}
        for split, split_list in target_prop_vals.items():
            for val in split_list:
                if val.compound_id not in target_val_dict[split]:
                    target_val_dict[split][val.compound_id] = val.target_val
                    in_range_dict[split][val.compound_id] = val.in_range

        if return_range:
            return target_val_dict, in_range_dict
        else:
            return target_val_dict

    def to_plot_df(self, agg_compounds=False, agg_losses=False, target_prop="pIC50"):
        """
        Convenience function for returning loss values in a DatFrame that can be used
        immediately for plotting.

        Parameters
        ----------
        agg_compounds : bool, default=False
            Aggregate (by mean) loss values for all compounds
        agg_losses : bool, default=False
            Aggregate (by weighted mean) all different types of loss values for each
            compound
        target_prop : str, default="pIC50"
            Target property to pull values for

        Returns
        -------
        pandas.DataFrame
            Plot-ready DataFrame
        """
        all_split = []
        all_epoch = []
        all_compounds = []
        all_preds = []
        all_target_vals = []
        all_range_vals = []
        all_losses = []
        all_loss_vals = []

        loss_dict = self.get_losses(agg_compounds=agg_compounds, agg_losses=agg_losses)
        preds_dict = self.get_predictions()
        target_vals_dict, in_range_dict = self.get_target_vals(
            target_prop, return_range=True
        )

        for sp, split_dict in loss_dict.items():
            match agg_compounds, agg_losses:
                case (False, False):
                    for compound_id, cpd_dict in split_dict.items():
                        preds = preds_dict[sp][compound_id]
                        target_val = target_vals_dict[sp][compound_id]
                        range_val = in_range_dict[sp][compound_id]
                        for loss_config, loss_val_list in cpd_dict.items():
                            all_split.extend([sp] * len(loss_val_list))
                            all_epoch.extend(np.arange(len(loss_val_list)))
                            all_compounds.extend([compound_id] * len(loss_val_list))
                            all_preds.extend(preds)
                            all_target_vals.extend([target_val] * len(loss_val_list))
                            all_range_vals.extend([range_val] * len(loss_val_list))
                            all_losses.extend([loss_config] * len(loss_val_list))
                            all_loss_vals.extend(loss_val_list)
                case (True, False):
                    for loss_config, loss_val_list in split_dict.items():
                        all_split.extend([sp] * len(loss_val_list))
                        all_epoch.extend(np.arange(len(loss_val_list)))
                        all_losses.extend([loss_config] * len(loss_val_list))
                        all_loss_vals.extend(loss_val_list)
                case (False, True):
                    for compound_id, loss_val_list in split_dict.items():
                        preds = preds_dict[sp][compound_id]
                        target_val = target_vals_dict[sp][compound_id]
                        range_val = in_range_dict[sp][compound_id]
                        all_split.extend([sp] * len(loss_val_list))
                        all_epoch.extend(np.arange(len(loss_val_list)))
                        all_compounds.extend([compound_id] * len(loss_val_list))
                        all_preds.extend(preds)
                        all_target_vals.extend([target_val] * len(loss_val_list))
                        all_range_vals.extend([range_val] * len(loss_val_list))
                        all_loss_vals.extend(loss_val_list)
                case (True, True):
                    loss_val_list = split_dict
                    all_split.extend([sp] * len(loss_val_list))
                    all_epoch.extend(np.arange(len(loss_val_list)))
                    all_loss_vals.extend(loss_val_list)

        use_vals = [
            (label, val)
            for label, val in zip(
                [
                    "split",
                    "epoch",
                    "compound_id",
                    "pred",
                    "target",
                    "in_range",
                    "loss_config",
                    "loss",
                ],
                [
                    all_split,
                    all_epoch,
                    all_compounds,
                    all_preds,
                    all_target_vals,
                    all_range_vals,
                    all_losses,
                    all_loss_vals,
                ],
            )
            if len(val) > 0
        ]

        return pandas.DataFrame(dict(use_vals))

    def to_loss_dict(self, allow_multiple=False, loss_config=None):
        """
        Method to convert a TrainingPredictionTracker to the old loss_dict method of
        tracking losses and predictions over training, for compatibility with existing
        methods.

        Parameters
        ----------
        allow_multiple : bool, default=False
            Allow multiple loss types for a given compound. If this is False, multiple
            loss types for a compound will raise a ValueError. If True, multiple loss
            types will generate an entry in the loss_dict for each loss type
        loss_config : LossFunctionConfig, optional
            Which LossFunctionConfig to pull values for
        """
        loss_dict = {"train": {}, "val": {}, "test": {}}
        # for sp, values_list in self.split_dict.items():
        for sp, values_list in self.get_values(loss_config=loss_config).items():
            for training_pred in values_list:
                if (training_pred.compound_id in loss_dict[sp]) and (
                    not allow_multiple
                ):
                    raise ValueError(
                        "Multiple loss config values found for compound",
                        training_pred.compound_id,
                    )

                d = {
                    "target": training_pred.target_val,
                    "in_range": training_pred.in_range,
                    "uncertainty": training_pred.uncertainty,
                    "preds": training_pred.predictions,
                    "losses": training_pred.loss_vals,
                    "pose_preds": training_pred.pose_predictions,
                }
                loss_dict[sp][training_pred.compound_id] = d

        return loss_dict

    def calculate_pred_statistics(self, target_prop, pred_epoch=-1):
        """
        Calculate boostrapped MAE, RMSE, Spearman r, and Kendall tau values on losses.
        Return value is a dict mapping split -> metric name -> {"value": metric value,
        "95ci_low": lower bound of bootstrapped 95% confidence interval,
        "95ci_high": upper bound of bootstrapped 95% confidence interval}.

        Parameters
        ----------
        target_prop : str
            Which target property to pull
        pred_epoch : int, default=-1
            Which epoch of predictions to use. Default is to use the final one

        Returns
        -------
        dict
            Statistics dict (see docstring for details)
        """
        # Get preds
        preds_dict = self.get_predictions(in_range_only=True)
        # preds_dict should map split -> compound_id -> preds (n_epochs,)

        # Get target vals
        target_vals = self.get_target_vals(
            target_prop=target_prop, flatten_compounds=False
        )
        # maps split -> compound_id -> target val

        # Order preds and targets together
        compound_ids_dict = {
            split: [compound_id for compound_id in split_d.keys()]
            for split, split_d in preds_dict.items()
        }  # maps split -> compound_ids
        preds_dict = {
            split: np.asarray(
                [
                    split_d[compound_id][pred_epoch]
                    for compound_id in compound_ids_dict[split]
                ]
            )
            for split, split_d in preds_dict.items()
        }  # maps split -> preds (n_compounds,) ordered by compound_ids_dict
        target_vals_dict = {
            split: np.asarray(
                [split_d[compound_id] for compound_id in compound_ids_dict[split]]
            )
            for split, split_d in target_vals.items()
        }  # maps split -> target_vals (n_compounds,) ordered by compound_ids_dict

        # Dict to accumulate results
        stats_dict = {"train": {}, "val": {}, "test": {}}

        # Loop through splits
        for sp in stats_dict.keys():
            preds = preds_dict[sp]
            target_vals = target_vals_dict[sp]

            # Calculate MAE and bootstrapped confidence interval
            mae = np.abs(target_vals - preds).mean()
            conf_interval = bootstrap(
                (target_vals, preds),
                statistic=lambda target, pred: np.abs(target - pred).mean(),
                method="basic",
                confidence_level=0.95,
                paired=True,
            ).confidence_interval
            stats_dict[sp]["mae"] = {
                "value": mae,
                "95ci_low": conf_interval.low,
                "95ci_high": conf_interval.high,
            }

            # Calculate RMSE and bootstrapped confidence interval
            rmse = np.sqrt(np.power(target_vals - preds, 2).mean())
            conf_interval = bootstrap(
                (target_vals, preds),
                statistic=lambda target, pred: np.sqrt(
                    np.power(target - pred, 2).mean()
                ),
                method="basic",
                confidence_level=0.95,
                paired=True,
            ).confidence_interval
            stats_dict[sp]["rmse"] = {
                "value": rmse,
                "95ci_low": conf_interval.low,
                "95ci_high": conf_interval.high,
            }

            # Calculate Spearman r and bootstrapped confidence interval
            sp_r = spearmanr(target_vals, preds).statistic
            conf_interval = bootstrap(
                (target_vals, preds),
                statistic=lambda target, pred: spearmanr(target, pred).statistic,
                method="basic",
                confidence_level=0.95,
                paired=True,
            ).confidence_interval
            stats_dict[sp]["sp_r"] = {
                "value": sp_r,
                "95ci_low": conf_interval.low,
                "95ci_high": conf_interval.high,
            }

            # Calculate Kendall's tau and bootstrapped confidence interval
            tau = kendalltau(target_vals, preds).statistic
            conf_interval = bootstrap(
                (target_vals, preds),
                statistic=lambda target, pred: kendalltau(target, pred).statistic,
                method="basic",
                confidence_level=0.95,
                paired=True,
            ).confidence_interval
            stats_dict[sp]["tau"] = {
                "value": tau,
                "95ci_low": conf_interval.low,
                "95ci_high": conf_interval.high,
            }

        return stats_dict


def _load_one_tpt(fn):
    print(fn, flush=True)
    return TrainingPredictionTracker(**json.loads(fn.read_text()))


def load_collection_tpt(
    top_level_dir: Path,
    model_dir_str: str,
    model_spec_kwargs: dict[str, list[str]],
    spec_name_to_output_name: dict[str, str] = None,
    spec_lab_to_output_lab: dict[str, dict[str, str]] = None,
    n_workers: int = 1,
):
    """
    Load a collection of TrainingPredictionTracker objects from a group of run
    directories that all share a similar name format. The output will be dict mapping
    from a dict giving the parameters used to load the run to the loaded
    TrainingPredictionTracker. The keys and values in the key tuple for the output can
    be transformed using the `spec_name_to_output_name` and `spec_lab_to_output_lab`
    parameters.

    Parameters
    ----------
    top_level_dir : Path
        The top level directory that contains all the desired run directories
    model_dir_str : str
        Format of all the run directories to access. This string will have the `format`
        method called on it, and all fields to be formatted should have names that match
        the keys in `model_spec_kwargs`
    model_spec_kwargs : dict[str, list[str]]
        Dict mapping list of field names from `model_dir_str` to all the possible
        options for each field. The runs that will be loaded will be the cartesian
        product of all the values in this dict
    spec_name_to_output_name : dict[str, str], optional
        Dict mapping the field name in `model_dir_str`/`model_spec_kwargs` to a
        different string to be used as the keys in the output key tuples. If not provided
        or if a specific field is missing, the raw field name will be used
    spec_lab_to_output_lab : dict[str, dict[str, str]], optional
        Dict mapping the field name in `model_dir_str`/`model_spec_kwargs` to a dict
        that provides mappings for the individual labels (ie the values in the lists
        in `model_spec_kwargs`) to a different string to be used as the values in the
        output key tuples. If not provided or if a specific field/label is missing, the
        raw values will be used
    n_workers : int, default=1
        Number of concurrent processes to use for loading files

    Returns
    -------
    dict[tuple[str, str], TrainingPredictionTracker]
    """

    # Gather all the files to load
    load_fns = []
    key_tups = []
    for spec_args in product(*model_spec_kwargs.values()):
        # Dict mapping field name to specific value for this iteration
        kwargs_dict = dict(zip(model_spec_kwargs.keys(), spec_args))

        # Format dir string and try to load files
        cur_model_dir = model_dir_str.format(**kwargs_dict)
        run_id_fn = top_level_dir / cur_model_dir / "run_id"
        if not run_id_fn.exists():
            print(kwargs_dict, "not run yet", flush=True)
            continue
        run_id = run_id_fn.read_text()
        pred_tracker_fn = top_level_dir / cur_model_dir / f"{run_id}/pred_tracker.json"
        if not pred_tracker_fn.exists():
            print(kwargs_dict, "still running", flush=True)
            continue
        load_fns.append(pred_tracker_fn)

        # Try and map field names and values
        key_dict = {}
        for orig_key, orig_val in kwargs_dict.items():
            # Use original key if it's not found in the mapping dict
            new_key = spec_name_to_output_name.get(orig_key, orig_key)
            # Try and get the val mapping dict, return an empty dict if not found so
            #  next step works
            val_map_dict = spec_lab_to_output_lab.get(orig_key, {})
            new_val = val_map_dict.get(orig_val, orig_val)
            key_dict[new_key] = new_val
        # Convert to tuple for hashing purposes
        key_tups.append(tuple(tuple(kvp) for kvp in key_dict.items()))

    with mp.Pool(processes=n_workers) as pool:
        res = pool.map(_load_one_tpt, load_fns)

    return dict(zip(key_tups, res))


def calc_epoch_stats(g):
    """
    Function to calculate per-epoch mean loss and MAE, meant to be called via df.apply.
    """
    return pandas.DataFrame(
        {"MAE": [np.mean(np.abs(g["pred"] - g["target"]))], "loss": [g["loss"].mean()]}
    )


def _load_one_df(fn, new_cols_dict, extract_epochs, target_prop):
    print(fn, flush=True)
    pred_tracker = TrainingPredictionTracker(**json.loads(fn.read_text()))

    # DF with each compound's pred for each epoch
    compound_df = pred_tracker.to_plot_df(
        agg_compounds=False, agg_losses=True, target_prop=target_prop
    )
    epoch_df = (
        compound_df.groupby(["split", "epoch"])
        .apply(calc_epoch_stats)
        .reset_index(level=["split", "epoch"])
        .reset_index(drop=True)
    )

    for new_key, new_val in new_cols_dict.items():
        # Set cols in dfs
        compound_df[new_key] = new_val
        epoch_df[new_key] = new_val

    # Will be a list of lists, so need to get the right list index
    per_compound_dfs = []
    for epoch in extract_epochs:
        if epoch == "all":
            per_compound_dfs.append(compound_df)
            continue
        if epoch == -1:
            epoch = compound_df["epoch"].max()
        elif epoch == "best_loss":
            idx = np.argmin(epoch_df["loss"])
            epoch = compound_df.iloc[idx, :]["epoch"]
        elif epoch == "best_mae":
            idx = np.argmin(epoch_df["MAE"])
            epoch = compound_df.iloc[idx, :]["epoch"]

        per_compound_dfs.append(compound_df.loc[compound_df["epoch"] == epoch, :])

    return epoch_df, per_compound_dfs


def load_collection_df(
    top_level_dir: Path,
    model_dir_str: str,
    model_spec_kwargs: dict[str, list[str]],
    spec_name_to_output_name: dict[str, str] = None,
    spec_lab_to_output_lab: dict[str, dict[str, str]] = None,
    extract_epochs: list[str | int] = None,
    target_prop: str = "pIC50",
    n_workers: int = 1,
):
    """
    Load a collection of TrainingPredictionTracker objects from a group of run
    directories that all share a similar name format. The output will be a DataFrame
    containing the loss and MAE for each epoch, as well as the per-compound predictions
    for each epoch specified in `extract_epochs`. Additionally a column will be added
    for each field in `model_spec_kwargs`, with the ability to map these variables to
    alternative names using the `spec_name_to_output_name` and `spec_lab_to_output_lab`
    parameters

    Parameters
    ----------
    top_level_dir : Path
        The top level directory that contains all the desired run directories
    model_dir_str : str
        Format of all the run directories to access. This string will have the `format`
        method called on it, and all fields to be formatted should have names that match
        the keys in `model_spec_kwargs`
    model_spec_kwargs : dict[str, list[str]]
        Dict mapping list of field names from `model_dir_str` to all the possible
        options for each field. The runs that will be loaded will be the cartesian
        product of all the values in this dict
    spec_name_to_output_name : dict[str, str], optional
        Dict mapping the field name in `model_dir_str`/`model_spec_kwargs` to a
        different string to be used as the keys in the output key tuples. If not provided
        or if a specific field is missing, the raw field name will be used
    spec_lab_to_output_lab : dict[str, dict[str, str]], optional
        Dict mapping the field name in `model_dir_str`/`model_spec_kwargs` to a dict
        that provides mappings for the individual labels (ie the values in the lists
        in `model_spec_kwargs`) to a different string to be used as the values in the
        output key tuples. If not provided or if a specific field/label is missing, the
        raw values will be used
    extract_epochs : list[str | int], optional
        Return per-compound predictions from specific epoch(s). A new DF will be created
        for each entry in this list. If the entry is an int, that int will be used as
        the epoch to extract. Special options for these entries are:
        * -1: take the last epoch
        * "all": return the per-compound predictions for every epoch (note that this DF
                 will likely be quite large)
        * "best_loss": take the epoch with the lowest loss
        * "best_mae": take the epoch with lowest MAE
    target_prop : str, default="pIC50"
        Target property to use when calling `pred_tracker.to_plot_df`
    n_workers : int, default=1
        Number of concurrent processes to use for loading files

    Returns
    -------
    dict[tuple[str, str], TrainingPredictionTracker]
    """
    if extract_epochs is None:
        extract_epochs = []
    mp_func = partial(
        _load_one_df, extract_epochs=extract_epochs, target_prop=target_prop
    )

    mp_args = []
    for spec_args in product(*model_spec_kwargs.values()):
        # Dict mapping field name to specific value for this iteration
        kwargs_dict = dict(zip(model_spec_kwargs.keys(), spec_args))

        # Format dir string and try to load files
        cur_model_dir = model_dir_str.format(**kwargs_dict)
        run_id_fn = top_level_dir / cur_model_dir / "run_id"
        if not run_id_fn.exists():
            print(kwargs_dict, "not run yet", flush=True)
            continue
        run_id = run_id_fn.read_text()
        pred_tracker_fn = top_level_dir / cur_model_dir / f"{run_id}/pred_tracker.json"
        if not pred_tracker_fn.exists():
            print(kwargs_dict, "still running", flush=True)
            continue

        new_cols_dict = {}
        for orig_key, orig_val in kwargs_dict.items():
            # Use original key if it's not found in the mapping dict
            new_key = spec_name_to_output_name.get(orig_key, orig_key)
            # Try and get the val mapping dict, return an empty dict if not found so
            #  next step works
            val_map_dict = spec_lab_to_output_lab.get(orig_key, {})
            new_val = val_map_dict.get(orig_val, orig_val)
            new_cols_dict[new_key] = new_val

        mp_args.append((pred_tracker_fn, new_cols_dict))

    with mp.Pool(processes=n_workers) as pool:
        res = pool.starmap(mp_func, mp_args)

    # Extract the results into lists of DFs to concatenate
    per_epoch_df = [r[0] for r in res]
    per_compound_dfs = list(zip(*[r[1] for r in res]))

    # Combine all lists of DFs into one DF
    per_epoch_df = pandas.concat(per_epoch_df, axis=0, ignore_index=True)
    per_compound_dfs = [
        pandas.concat(compound_df, axis=0, ignore_index=True)
        for compound_df in per_compound_dfs
    ]

    return per_epoch_df, per_compound_dfs
