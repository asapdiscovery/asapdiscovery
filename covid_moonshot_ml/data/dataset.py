from Bio.PDB.PDBParser import PDBParser
import numpy as np
from rdkit.Chem import GetPeriodicTable
import torch
from torch.utils.data import Dataset

class DockedDataset(Dataset):
    """
    Class for loading docking results into a dataset to be used for graph
    learning.
    """
    def __init__(self, str_fns, compounds, ignore_h=True, lig_resn='LIG'):
        """
        Parameters
        ----------
        str_fns : List[str]
            List of paths for the PDB files. Should correspond 1:1 with the
            names in compounds.
        compounds : List[Tuple[str]]
            List of (crystal structure, ligand compound id).
        """

        super(DockedDataset, self).__init__()

        self.compounds = compounds

        table = GetPeriodicTable()
        self.structures = {}
        for fn, compound in zip(str_fns, compounds):
            # pdb_str = Chem.MolFromPDBFile(fn)
            # if pdb_str is None:
            #     print(fn, flush=True)

            # conf = pdb_str.GetConformer()
            # atomic_nums = [a.GetAtomicNum() for a in pdb_str.GetAtoms()]
            # atom_pos = [list(conf.GetAtomPosition(i)) \
            #     for i in range(len(atomic_nums))]

            s = PDBParser().get_structure(f'{compound[0]}_{compound[1]}', fn)
            ## Filter out water residues
            all_atoms = [a for a in s.get_atoms() if a.parent.resname != 'HOH']
            if ignore_h:
                all_atoms = [a for a in all_atoms if a.element != 'H']
            ## Fix multi-letter atom elements being in all caps (eg CL)
            atomic_nums = [table.GetAtomicNumber(a.element.title()) \
                for a in all_atoms]
            atom_pos = [tuple(a.get_vector()) for a in all_atoms]
            is_lig = [a.parent.resname == lig_resn for a in all_atoms]

            self.structures[compound] = {
                'z': torch.tensor(atomic_nums),
                'pos': torch.tensor(atom_pos).float(),
                'lig': torch.tensor(is_lig)
            }

    def __len__(self):
        return(len(self.compounds))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        if (type(idx) is int) or (type(idx) is tuple):
            return_list = False
            idx_type = type(idx)
            idx = [idx]
        else:
            return_list = True
            if type(idx[0]) is int:
                idx_type = int
            else:
                idx_type = tuple
                idx = [tuple(i) for i in idx]

        ## If idx is integral, assume it is indexing the structures list,
        ##  otherwise assume it's giving structure name
        if idx_type is int:
            compounds = [self.compounds[i] for i in idx]
        else:
            compounds = idx

        compounds = [c for c in compounds if c in self.structures]
        str_list = [self.structures[c] for c in compounds]
        if return_list:
            return(compounds, str_list)
        else:
            return(compounds[0], str_list[0])

    def __iter__(self):
        for s in self.compounds:
            yield self[s]