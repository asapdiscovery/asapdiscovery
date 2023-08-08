from collections import namedtuple
from openmm.app import PME, HBonds
from openmm.unit import amu, nanometers

ForceFieldParams = namedtuple(
    "ForceFieldParams",
    [
        "ff_xmls",
        "padding",
        "water_model",
        "nonbonded_method",
        "nonbonded_cutoff",
        "ewald_error_tolerance",
        "constraints",
        "rigid_water",
        "hydrogen_mass",
    ],
)


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


