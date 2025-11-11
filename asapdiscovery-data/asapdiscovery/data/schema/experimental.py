from datetime import date
from typing import Any

from pydantic.v1 import BaseModel, Field


class ExperimentalCompoundData(BaseModel):
    compound_id: str = Field(
        None,
        description="The unique compound identifier (PostEra or enumerated ID)",
    )

    smiles: str = Field(
        None,
        description="OpenEye canonical isomeric SMILES string defining suspected SMILES of racemic mixture (with unspecified stereochemistry) or specific enantiopure compound (if racemic=False); may differ from what is registered under compound_id.",
    )

    racemic: bool = Field(
        False,
        description="If True, this experiment was performed on a racemate; if False, the compound was enantiopure.",
    )

    achiral: bool = Field(
        False,
        description="If True, this compound has no chiral centers or bonds, by definition enantiopure",
    )

    absolute_stereochemistry_enantiomerically_pure: bool = Field(
        False,
        description="If True, the compound was enantiopure and stereochemistry recorded in SMILES is correct",
    )

    relative_stereochemistry_enantiomerically_pure: bool = Field(
        False,
        description="If True, the compound was enantiopure, but unknown if stereochemistry recorded in SMILES is correct",
    )

    date_created: date = Field(None, description="Date the molecule was created.")

    experimental_data: dict[str, float | Any] = Field(
        dict(),
        description='Experimental data fields, including "pIC50" and uncertainty (either "pIC50_stderr" or  "pIC50_{lower|upper}"',
    )

    def to_SD_tags(self) -> tuple[dict[str, str], dict[str, float]]:
        """
        Convert to a dictionary of SD tags
        """
        data = self.dict()
        exp_data = data.pop("experimental_data")
        # cannot use a nested dict in SD tags, so flatten to two tags
        data = {str(k): str(v) for k, v in data.items() if v is not None}
        exp_data = {str(k): float(v) for k, v in exp_data.items() if v is not None}
        return data, exp_data

    class Config:
        allow_mutation = False
        extra = "forbid"
