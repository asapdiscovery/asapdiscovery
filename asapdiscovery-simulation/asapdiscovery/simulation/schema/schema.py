from collections import namedtuple

from openmm.app import PME, HBonds
from openmm.unit import amu, nanometers

ForceFieldParams = namedtuple(
    "ForceFieldParams",
    [
        "ff_xmls",
        "padding",
        "model",
        "nonbondedMethod",
        "nonbondedCutoff",
        "ewaldErrorTolerance",
        "constraints",
        "rigidWater",
        "hydrogenMass",
    ],
)


DefaultForceFieldParams = ForceFieldParams(
    ff_xmls=[
        "amber14-all.xml",
        "amber14/tip3pfb.xml",
    ],
    padding=0.9 * nanometers,
    model="tip3p",
    nonbondedMethod=PME,
    nonbondedCutoff=1.0 * nanometers,
    ewaldErrorTolerance=0.00001,
    constraints=HBonds,
    rigidWater=True,
    hydrogenMass=4.0 * amu,
)
