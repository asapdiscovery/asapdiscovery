# crossdocking.py
from asapdiscovery.docking import DockingInputs, DockOE
from asapdiscovery.docking.components import Edge
from asapdiscovery.docking.utils import Multiprocessor


# Load Docking inputs file and get data representations of the components
docking_inputs = DockingInputs(args.docking_inputs)
ligands, targets = docking_inputs.GetComponents()

# Load Docking options and get function to pass to multiprocessing
docking_obj = DockOE(args.docking_options)
logger.info(f"Using docking options: {docking_obj.GetOptions()}")

# This is probably just an internal partial(function, self.options) call
docking_function = docking_obj.GetFunction(output_dir=args.output_dir)

# For cross-docking, docking everything to everything else, except itself
edges = [Edge(ligand, target) for ligand, target in zip(ligands, targets) if not ligand.source == target.source]

if debug_num:
    edges = edges[:debug_num]

# Prepare to run the docking objects
multiprocessor = Multiprocessor(func=docking_function,
                                arguments=edges,
                                n=args.num_nodes,
                                timeout=args.timeout,
                                max_failures=args.max_failures,
                                log_name = args.log_name)

multiprocessor.run()
###########################################################################################
# Calculate MCS similarity between query ligands and target complexes
from asapdiscovery.data.openeye import load_openeye_sdfs, oechem
from asapdiscovery.data.database import ReadIDs, DataQuery, WriteIDs
from asapdiscovery.analysis.ligand import CalculateSimilarity
from asapdiscovery.docking import DockingInputs, DockOE

# Class that can read model ids from txt files, sdfs, or csvs (?)
query_ligand_ids = ReadIDS(args.query_ligands)
target_complex_ids = ReadIDS(args.target_complexes)

# Returns Generator
query_ligands = DataQuery(query_ligand_ids)
target_complexes = DataQuery(target_complex_ids)


def similarity_func(query_ligand, target_complexes):
    reference_ligands = [target_complex.ligand for target_complex in target_complexes]

    similarities = CalculateSimilarity(query_ligand, reference_ligands, metric="MCSS")

    similarities.serialize(args.output_dir)

# There's a smarter way to do this with itertools.combinations where you only have to compute half of the comparisons
# But then I'm not sure how you parallelize efficiently
comparisons = [(query_ligand, target_complexes) for query_ligand in query_ligands]

# Multiprocessing functions handles various option
multiprocessor = Multiprocessor(func=similarity_func,
                                arguments=comparisons,
                                n=args.num_nodes,
                                timeout=args.timeout,
                                max_failures=args.max_failures,
                                log_name = args.log_name)

multiprocessor.run()

# Probably want a function to concatenate the similarity scores (?)

##########################################################################################
# asapdiscovery.analysis.ligand

def CalculateSimilarity(query_ligand, reference_ligands, metric="MCSS"):
    query_mol = query_ligand.load_as(oechem.OEMol)
    reference_mols = [reference_ligand.load_as(oechem.OEMol) for reference_ligand in reference_ligands]

    if metric == "MCSS":
        similarity_scores = CalculateMCSS(query_mol, reference_mols)

    return similarity_scores

#########################################################################################
# asapdiscovery.analysis

class Scores():
    def serialize(self, output_dir):
        pass
    def deserialize(self, output_dir):
        pass

class PairwiseScores(Scores):
    def __init__(self, scores):
    def get_top_n(self, query, n):
        # returns model ids of top n scores for a given query


###########################################################################################
# MCSS-Based Docking
from asapdiscovery.analysis import PairwiseScores

# Load similarity scores
# could also load a list of serialized similarity scores
similarities = PairwiseScores(args.similarity_scores)

# Load Docking options and get function to pass to multiprocessing
docking_obj = DockOE(args.docking_options)
logger.info(f"Using docking options: {docking_obj.GetOptions()}")

# This is probably just an internal partial(function, self.options) call
docking_function = docking_obj.GetFunction(output_dir=args.output_dir)

ligands = ReadIDs(args.ligands)

# Construct edges
edges = [Edge(ligand, complex) for ligand in ligands for complex in
         DataQuery(similarities.get_top_n(ligand.model_id, args.top_n))]

# run multiprocessing, etc




############################################################################################
from asapdiscovery.data.database import DataQuery
from pathlib import Path

class DockingInputs():

    def __init__(self, docking_inputs_file=None, ligand_ids=None, target_ids=None):
        if docking_inputs_file:
            self.deserialize(docking_inputs_file)
        elif ligand_ids and target_ids:
            self.ligand_ids = ligand_ids
            self.target_ids = target_ids
        else:
            raise ValueError("Must provide either docking_inputs_file or ligand_ids and target_ids")

    def GetComponents(self):
        ligands = DataQuery(self.ligand_ids())
        targets = DataQuery(self.target_ids())

        return ligands, targets

    def deserialize(self, output_dir):
        # deserialize DockingInputs
        # could be as simple as loading a .csv file and saving the dataframe to the object

        # for example
        import pandas as pd
        df = pd.read_csv(output_dir)
        self.ligand_ids = df['Ligand_ID']
        self.target_ids = df['Target_ID']
        self.df = df

    def serialize(self, output_dir: Path):
        # serialize DockingInputs
        import pandas as pd
        self.df.to_csv(output_dir: Path)


###############################################################################################
# asapdiscovery.data.database
from typing import Union
from asapdiscovery.data.schema import Ligand, Target, Complex

def GetLocalDB(self):
    # checks to make sure local database is not corrupted

    # if not corrupted, returns local database

    # probably should do this with some kind of generator

    return db

db = GetLocalDB()

def DataQuery(self, model_ids: list[str]):
    for model_id in model_ids:
        if model_id in db:
            model_data = db.get(model_id)

            if model_data.type in Ligand.types:
                yield Ligand(model_data)
            elif model_data.type in Target.types:
                yield Target(model_data)
            elif model_data.type in Complex.types:
                yield Complex(model_data)
            else:
                raise ValueError(f"{model_id} with type '{model_data.type}' is not a valid model type")
        else:
            raise ValueError(f"{model_id} not found in local database")

#################################################################################################
# asapdiscovery.data.schema
from typing import Union
from pydantic import Field, BaseModel

class Model(BaseModel):
    model_id: str = Field(..., description="ID of model")

    id: str = Field(..., description="ID of molecule represented by model. i.e. Compound_ID, Fragalysis Dataset, etc")

    type: str = Field(..., description=("Type of model)

    path: Path = Field(..., description="Path to model file")

    format: str = Field(..., description="Format of model file (pdb, sdf, mol2, smiles, etc.)")

    source: str = Field(..., description="Source of model (e.g. PDB ID, SMILES string, fragalysis dataset, etc.)")

    types: list[str] = ("xtal_ligand", "xtal_target", "xtal_complex", "docked_ligand", "docked_target",
                        "docked_complex", "2d_ligand", "smiles_ligand")

    formats: list[str] = ("pdb", "sdf", "mol2", "smiles")

    def __init__(self, model_data):
        self.constructor(model_data)

    def constructor(self, model_data):
        # Some function that converts database entries into attributes of this class
        if not model_data.type in self.types:
            raise ValueError(f"{model_data.type} is not a valid model type")

    def load_as(self, object_type):
        # Some function that loads the model file as the specified format
        # probably need a better way to construct this than a fully enumerated if statement for each type and format
        if type(object_type) == oechem.OEMol and self.format == "sdf":
            from asapdiscovery.data.openeye import load_openeye_sdf
            return load_openeye_sdf(self.path)

        # also would want a way to check to make sure this object actually includes a smiles string
        elif type(object_type) == oechem.OEMol and self.format == "smiles":
            from asapdiscovery.data.openeye import oechem
            mol = oechem.OEMol()
            if oechem.OESmilesToMol(mol, self.smiles):
                return mol
            else:
                raise ValueError(f"Unable to construct OpenEye molecule from {self.smiles}")

class Ligand(Model):
    # Here we define the fields that are required for a ligand
    types: list[str] = ["xtal_ligand", "docked_ligand", "2d_ligand", "smiles_ligand"]
    formats: list[str] = ["sdf", "mol2", "smiles"]
    smiles: str = Field(..., description="SMILES string of ligand")
    complex_id = Field(..., description="ID of complex that ligand is part of")

    def get_complex(self):
        if not self.complex_id:
            raise ValueError("Ligand does not have a complex_id")
        return Complex(self.complex_id)

class Complex(Model):
    def __init__(self, model_data):
        self.constructor(model_data)
        self.ligand = Ligand(self.ligand_id)
        self.target = Target(self.target_id)

