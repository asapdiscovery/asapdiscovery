from pydantic import BaseModel, Field

from asapdiscovery.ml.config import LossFunctionConfig


class TrainingPrediction(BaseModel):
    """
    Class to organize predictions made during training.
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
