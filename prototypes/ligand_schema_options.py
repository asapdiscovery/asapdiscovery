# Composition over Inheritance
# Should we have a single Ligand class with optional attributes?
# which are validated manually in every individual function?
# or should we have a base Ligand class and then subclasses for each set of ligand data?
from pydantic import Field, BaseModel
from datetime import date
from typing import Optional

############### OPTION 1: Composition ############################
# Ligand class stores all the core info
# Data classes (LigandExperimentalData, LigandCrystalData) store the other stuff
# Then you can compose a LigandData object from ligand and data classes
# Validation is a bit more annoying bc you can't rely on any classes to for sure have the info you need


class ProvenanceBase(BaseModel):
    pass


class Ligand(BaseModel):
    id: Optional[str]
    vc_id_postera: Optional[str]
    moonshot_compound_id: Optional[str]
    target_id: Optional[str]
    source = str
    smiles: str
    ligand_provenance: Optional[ProvenanceBase]


# copied from asapdiscovery-data/asapdiscovery/data/schema.py
class LigandExperimentalData(BaseModel):
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

    experimental_data: dict[str, float] = Field(
        dict(),
        description='Experimental data fields, including "pIC50" and uncertainty (either "pIC50_stderr" or  "pIC50_{lower|upper}"',
    )


class LigandCrystalData(BaseModel):
    crystal_id: Optional[str]
    crystal_provenance: Optional[ProvenanceBase]
    dataset: str = Field(
        None, description="Dataset name from Fragalysis (name of structure)."
    )

    str_fn: str = Field(None, description="Filename of the PDB structure.")

    sdf_fn: str = Field(None, description="Filename of the SDF file")
    active_site_chain: str = Field(
        None, description="Chain identifying the active site of interest."
    )


class LigandData(BaseModel):
    ligand: Ligand
    experimental_data: Optional[LigandExperimentalData]
    crystal_data: Optional[LigandCrystalData]


# Example usage (pls ignore the chemistry)
ligand1 = Ligand(id="1", smiles="CCO", source="postera")
ligand2 = Ligand(id="2", smiles="CCCO", source="postera")

ligand1_exp_data = LigandExperimentalData(racemic=True, achiral=True)
ligand2_exp_data = LigandExperimentalData(racemic=False, achiral=False)

ligand1_crystal_data = LigandCrystalData(crystal_id="1", dataset="DS1")

ligand1_data = LigandData(
    ligand=ligand1,
    experimental_data=ligand1_exp_data,
    crystal_data=ligand1_crystal_data,
)
ligand1_data_crystal_only = LigandData(
    ligand=ligand1, crystal_data=ligand1_crystal_data
)
ligand2_data = LigandData(ligand=ligand2, experimental_data=ligand2_exp_data)

if ligand1 == ligand2:
    raise ValueError("These definitely shouldn't be equal")

if ligand1_data == ligand1_data_crystal_only:
    raise NotImplementedError("Not sure if this should return a value error or not")

if ligand1_data.ligand != ligand1_data_crystal_only.ligand:
    raise ValueError("I think these should be equal")


def simple_function(ligand_data: LigandData):
    print(ligand_data.ligand.smiles)


def function_that_requires_crystal_data(ligand_data: LigandData):
    # Now we have to put this at the beginning of every function like this?
    # instead of relying on the class to have the data we need?
    if ligand_data.crystal_data is None:
        raise ValueError("This function requires crystal data")


simple_function(ligand1_data)
simple_function(ligand2_data)

function_that_requires_crystal_data(ligand1_data)
function_that_requires_crystal_data(ligand2_data)
