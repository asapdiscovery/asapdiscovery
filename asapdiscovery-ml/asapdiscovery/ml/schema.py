from pydantic import BaseModel, Extra, Field, validator

from asapdiscovery.ml.config import LossFunctionConfig


class TrainingPrediction(BaseModel):
    """
    Class to store predictions made during training.
    """

    # Identifiers
    compound_id: str = Field(..., description="Compound ID for the ligand.")
    xtal_id: str = Field(..., description="Crystal ID for the protein structure.")

    # Target info
    target_prop: str = Field(..., description="Target property being predicted.")
    target_val: float = Field(..., description="Target value to predict.")
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

    class Config:
        # Allow things to be added to the object after initialization/validation
        extra = Extra.allow


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

    @validator("split_dict")
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
            return (compound_id is None) or (query == compound_id)

        def xtal_id_match(query):
            return (xtal_id is None) or (query == xtal_id)

        def target_prop_match(query):
            return (target_prop is None) or (query == target_prop)

        def loss_config_match(query):
            return (loss_config is None) or (query == loss_config)

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
            return [self.split_dict[split][i] for i in return_idxs]
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
    ):
        """
        Get TrainingPrediction values based on passed filters. The type of the return
        value will depend on the filters passed. If split is not passed, the result will
        be a dict, giving a mapping of split: list[TrainingPrediction]. If a split is
        given, then a list of the TrainingPredictions found in that split will be
        returned.

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
            print(
                "No matches found for",
                ", ".join([split, compound_id, xtal_id, target_prop, str(loss_config)]),
            )

        # Check that we've only got one, if necessary
        if not allow_multiple:
            if num_found > 1:
                raise ValueError(
                    "Multiple results found for search "
                    ", ".join(
                        [split, compound_id, xtal_id, target_prop, str(loss_config)]
                    )
                )

        for sp, split_list in self.split_dict.items():
            for i in return_idxs[sp]:
                split_list[i].predictions.append(prediction)
                split_list[i].pose_predictions.append(pose_predictions)
                split_list[i].loss_vals.append(loss_val)
