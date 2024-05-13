from pydantic import BaseModel, Field, validator

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
    predictions: list[float] = Field(..., description="Model prediction.")
    pose_predictions: list[list[float]] = Field(
        ...,
        description="Single-pose model prediction for each pose in input.",
    )

    # Loss info
    loss_config: LossFunctionConfig = Field(
        ..., description="Config describing loss function."
    )
    loss_vals: list[float] = Field(..., description="Loss value of model prediction.")


class TrainingPredictionTracker(BaseModel):
    """
    Class to organize predictions.
    """

    split_dict: dict[str, list[TrainingPrediction]] = Field(
        None, description="Internal dict storing all TrainingPredictions."
    )

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
