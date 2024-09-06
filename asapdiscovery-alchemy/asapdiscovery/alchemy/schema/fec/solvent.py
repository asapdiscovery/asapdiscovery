import abc
from typing import Literal

import gufe
from asapdiscovery.alchemy.schema.fec.base import _SchemaBase
from openff.models.types import FloatQuantity
from openff.units import unit as OFFUnit
from pydantic import Field


class SolventSettings(_SchemaBase):
    """
    A settings class to encode the solvent used in the OpenFE FEC calculations.
    """

    type: Literal["SolventSettings"] = "SolventSettings"

    smiles: str = Field("O", description="The smiles pattern of the solvent.")
    positive_ion: str = Field(
        "Na+",
        description="The positive monoatomic ion which should be used to neutralize the system and to adjust the ionic concentration.",
    )
    negative_ion: str = Field(
        "Cl-",
        description="The negative monoatomic ion which should be used to neutralize the system and to adjust the ionic concentration.",
    )
    neutralize: bool = Field(
        True,
        description="If the net charge of the chemical system should be neutralized by the ions defined by `positive_ion` and `negative_ion`.",
    )
    ion_concentration: FloatQuantity["molar"] = Field(  # noqa: F821
        0.15 * OFFUnit.molar,
        description="The ionic concentration required in molar units.",
    )

    def to_solvent_component(self) -> gufe.SolventComponent:
        return gufe.SolventComponent(**self.dict(exclude={"type"}))
