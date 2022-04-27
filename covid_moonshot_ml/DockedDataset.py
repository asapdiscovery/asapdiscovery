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
        str_fns : list[str]
            List of paths for the PDB files. Should correspond 1:1 with the
            names in compounds
        compounds : list[tuple[str]]
            List of (crystal structure, ligand compound id)
        ignore_h : bool, default=True
            Whether to remove hydrogens from the loaded structure
        lig_resn : str, default='LIG'
            The residue name for the ligand molecules in the PDB files. Used to
            identify which atoms belong to the ligand
        """

        super(DockedDataset, self).__init__()

        self.compounds = compounds

        table = GetPeriodicTable()
        self.structures = {}
        for fn, compound in zip(str_fns, compounds):
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
        """
        Parameters
        ----------
        idx
            Index into dataset. Can either be a numerical index into the
            structures or a tuple of (crystal structure, ligand compound id),
            or a list/torch.tensor/numpy.ndarray of either of those types
        """
        ## Extract idx from inside the tensor object
        if torch.is_tensor(idx):
            try:
                idx = idx.item()
            except ValueError:
                idx = idx.tolist()

        ## Figure out the type of the index, and keep note of whether a list was
        ##  passed in or not
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