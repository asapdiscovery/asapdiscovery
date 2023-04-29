# crossdocking.py
from asapdiscovery.docking import DockingInputs, DockOE

docking_inputs = DockingInputs(args.docking_input_path)

ligands, targets = docking_inputs.GetComponents()

# For cross-docking, docking everything to everything else, except itself
edges = [(ligand, target) for ligand, target in zip(ligands, targets) if not ligand.source == target.source]






############################################################################################
from asapdiscovery.data.database import DataQuery
from asapdiscovery.docking.components import Ligand, Target
from pathlib import Path

class DockingInputs():

    def __init__(self, docking_inputs_file):
        self.deserialize(docking_inputs_file)

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

def GetLocalDB(self):

    return db

db = GetLocalDB()

def DataQuery(self, model_ids: Union(str, list[str])):
    model_id
