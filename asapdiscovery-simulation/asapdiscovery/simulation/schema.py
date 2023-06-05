from pydantic import BaseModel, Field

from openmm.app import PME, HBonds
from openmm.unit import amu, nanometers


class ForceFieldParams(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    ff_xmls: list[str] = Field(description="List of forcefield xmls to use.")
    padding = Field(description="Padding to add to box size, i.e. 0.9 * nanometers.")
    water_model: str = Field(description="Water model to use, i.e. tip3p.")
    nonbonded_method = Field(description="Nonbonded method to use, i.e. PME.")
    nonbonded_cutoff = Field(description="Nonbonded cutoff, i.e. 1.0 * nanometers.")
    ewald_error_tolerance: float = Field(
        description="Ewald error tolerance. i.e. 10^-5."
    )
    constraints = Field(description="Constraints to use (i.e. HBonds)")
    rigid_water: bool = Field(description="Whether to use a rigid water model.")
    hydrogen_mass = Field(description="Hydrogen mass, i.e. 4.0 * amu.")


DefaultForceFieldParams = ForceFieldParams(
    ff_xmls=[
        "amber14-all.xml",
        "amber14/tip3pfb.xml",
    ],
    padding=0.9 * nanometers,
    water_model="tip3p",
    nonbonded_method=PME,
    nonbonded_cutoff=1.0 * nanometers,
    ewald_error_tolerance=0.00001,
    constraints=HBonds,
    rigid_water=True,
    hydrogen_mass=4.0 * amu,
)
