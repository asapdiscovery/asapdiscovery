from typing import List, Optional
from asapdiscovery.data.schema import Ligand, Target
from asapdiscovery.docking.schema import DockingInputs, DockingOpts, Docker_OE, LigandPrepper, ProteinPrepper


ligand_fr = FileReader(ligand_fn, type="ligand", format='sdf')
ligands: List[Ligand] = ligand_fr.load()

lp = LigandPrepper(ligands)
prepped_ligands = lp.prep()

target_fr = FileReader(target_fn, type="target", format='pdb')

targets: List[Target] = target_fr.load()

pp = ProteinPrepper(targets)
prepped_targets = pp.prep()

ligand_selector = MCSSelector(prepped_targets, prepped_ligands)
docking_inputs: DockingInputs = ligand_selector.select(top_n)

docking_opts = DockingOpts(docking_opts_fn)
docker = Docker_OE(docking_inputs, docking_opts)

docker.dock()

dr.to_csv()
dr.to_disc()

#######################################################################################################################
from asapdiscovery.data.openeye import load_openeye_sdfs, oechem

class FileReader():
    def __init__(self, fn, type, format):
        self.fn = fn
        self.type = type
        self.format = format

    def check_file_exists(self):
        pass


    def load(self):
        is self.typw=="ligand":
            return self.load_ligand()
        elif self.type=="target":
            return self.load_target()



    def load_ligand(self):
        if self.format = smi:
            mols = load_openeye_smi(self.fn)
        elif self.format = sdf:
            mols = load_openeye_sdf(self.fn)
        else:
            raise ValueError(f"Format {self.format} not supported")

        lignands = [Ligand.from_oemol(mol) for mol in mols]
        return lignands

    def load_target(self):
        pass

###################################################

ref_ligands = [target.get_ligand() for target in targets]
