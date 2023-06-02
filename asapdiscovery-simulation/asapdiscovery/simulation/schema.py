from collections import namedtuple

from openmm.app import PME, ForceField, HBonds, Modeller, PDBFile
from openmm.unit import amu, nanometers
from pydantic import BaseModel, Field

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
    [
        "amber14-all.xml",
        "amber14/tip3pfb.xml",
    ],
    0.9 * nanometers,
    "tip3p",
    PME,
    1.0 * nanometers,
    0.000001,
    HBonds,
    True,
    4.0 * amu,
)
