from torch.utils.data import Dataset


class DockedDataset(Dataset):
    """
    Class for loading docking results into a dataset to be used for graph
    learning.
    """

    def __init__(
        self,
        str_fns,
        compounds,
        ignore_h=True,
        lig_resn="LIG",
        extra_dict=None,
    ):
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
        extra_dict : dict[str, dict], optional
            Extra information to add to each structure. Keys should be
            compounds, and dicts can be anything as long as they don't have the
            keys ["z", "pos", "lig", "compound"]
        """
        from Bio.PDB.PDBParser import PDBParser
        from rdkit.Chem import GetPeriodicTable
        import torch

        super(DockedDataset, self).__init__()

        table = GetPeriodicTable()
        self.compounds = {}
        self.structures = []
        for i, (fn, compound) in enumerate(zip(str_fns, compounds)):
            s = PDBParser().get_structure(f"{compound[0]}_{compound[1]}", fn)
            ## Filter out water residues
            all_atoms = [a for a in s.get_atoms() if a.parent.resname != "HOH"]
            if ignore_h:
                all_atoms = [a for a in all_atoms if a.element != "H"]
            ## Fix multi-letter atom elements being in all caps (eg CL)
            atomic_nums = [
                table.GetAtomicNumber(a.element.title()) for a in all_atoms
            ]
            atom_pos = [tuple(a.get_vector()) for a in all_atoms]
            is_lig = [a.parent.resname == lig_resn for a in all_atoms]

            try:
                self.compounds[compound].append(i)
            except KeyError:
                self.compounds[compound] = [i]

            ## Create new structure and update from extra_dict
            new_structure = {
                "z": torch.tensor(atomic_nums),
                "pos": torch.tensor(atom_pos).float(),
                "lig": torch.tensor(is_lig),
                "compound": compound,
            }
            if extra_dict and (compound in extra_dict):
                new_structure.update(extra_dict[compound])
            self.structures.append(new_structure)

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int, tuple, list[tuple/int], tensor[tuple/int]
            Index into dataset. Can either be a numerical index into the
            structures or a tuple of (crystal structure, ligand compound id),
            or a list/torch.tensor/numpy.ndarray of either of those types

        Returns
        -------
        list[tuple]
            List of tuples (crystal_structure, compound_id) for found structures
        list[dict]
            List of dictionaries with keys
            - `z`: atomic numbers
            - `pos`: position matrix
            - `lig`: ligand identifier
            - `compound`: tuple of (crystal_structure, compound_id)
        """
        import torch

        ## Extract idx from inside the tensor object
        if torch.is_tensor(idx):
            try:
                idx = idx.item()
            except ValueError:
                idx = idx.tolist()

        ## Figure out the type of the index, and keep note of whether a list was
        ##  passed in or not
        if type(idx) is int:
            return_list = False
            idx_type = int
            idx = [idx]
        else:
            return_list = True
            if type(idx[0]) is int:
                idx_type = int
            else:
                idx_type = tuple
                if (
                    (type(idx) is tuple)
                    and (len(idx) == 2)
                    and (type(idx[0]) is str)
                    and (type(idx[1]) is str)
                ):
                    idx = [idx]
                else:
                    idx = [tuple(i) for i in idx]

        ## If idx is integral, assume it is indexing the structures list,
        ##  otherwise assume it's giving structure name
        if idx_type is int:
            str_idx_list = idx
        else:
            ## Need to find the structures that correspond to this compound(s)
            str_idx_list = [i for c in idx for i in self.compounds[c]]

        str_list = [self.structures[i] for i in str_idx_list]
        compounds = [s["compound"] for s in str_list]
        if return_list:
            return (compounds, str_list)
        else:
            return (compounds[0], str_list[0])

    def __iter__(self):
        for s in self.structures:
            yield (s["compound"], s)


### TODO before PR: fix compound_id_dict here to match with new format
class GraphDataset(Dataset):
    """
    Class for loading SMILES as graphs.
    """

    def __init__(
        self,
        exp_compounds,
        compound_id_dict=None,
        node_featurizer=None,
        edge_featurizer=None,
        cache_file=None,
    ):
        """
        Parameters
        ----------
        exp_compounds : List[schema.ExperimentalCompoundData]
            List of compounds
        compound_id_dict : Dict[str, str], optional
            Dict mapping from compound_id to Mpro dataset
        node_featurizer : BaseAtomFeaturizer, optional
            Featurizer for node data
        edge_featurizer : BaseBondFeaturizer, optional
            Featurizer for edges
        cache_file : str, optional
            Cache file for graph dataset

        """
        from dgllife.data import MoleculeCSVDataset
        from dgllife.utils import SMILESToBigraph
        import pandas

        ## Build dataframe
        all_compound_ids, all_smiles, all_pic50 = zip(
            *[
                (c.compound_id, c.smiles, c.experimental_data["pIC50"])
                for c in exp_compounds
            ]
        )
        df = pandas.DataFrame(
            {
                "compound_id": all_compound_ids,
                "smiles": all_smiles,
                "pic50": all_pic50,
            }
        )

        ## Build dataset
        smiles_to_g = SMILESToBigraph(
            add_self_loop=True,
            node_featurizer=node_featurizer,
            edge_featurizer=edge_featurizer,
        )
        dataset = MoleculeCSVDataset(
            df=df,
            smiles_to_graph=smiles_to_g,
            smiles_column="smiles",
            cache_file_path=cache_file,
            task_names=["pic50"],
        )

        self.compounds = {}
        self.structures = []
        for i, (compound_id, g) in enumerate(zip(all_compound_ids, dataset)):
            ## Make compound tuple
            if compound_id_dict:
                compound = (
                    compound_id_dict.get(compound_id, "NA"),
                    compound_id,
                )
            else:
                compound = ("NA", compound_id)

            ## Add data
            try:
                self.compounds[compound].append(i)
            except KeyError:
                self.compounds[compound] = [i]
            self.structures.append(
                {"smiles": g[0], "g": g[1], "pic50": g[2], "compound": compound}
            )

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int, tuple, list[tuple/int], tensor[tuple/int]
            Index into dataset. Can either be a numerical index into the
            structures or a tuple of (crystal structure, ligand compound id),
            or a list/torch.tensor/numpy.ndarray of either of those types

        Returns
        -------
        list[tuple]
            List of tuples (crystal_structure, compound_id) for found structures
        list[dict]
            List of dictionaries with keys
            - `g`: DGLGraph
            - `compound`: tuple of (crystal_structure, compound_id)
        """
        import torch

        ## Extract idx from inside the tensor object
        if torch.is_tensor(idx):
            try:
                idx = idx.item()
            except ValueError:
                idx = idx.tolist()

        ## Figure out the type of the index, and keep note of whether a list was
        ##  passed in or not
        if type(idx) is int:
            return_list = False
            idx_type = int
            idx = [idx]
        else:
            return_list = True
            if type(idx[0]) is int:
                idx_type = int
            else:
                idx_type = tuple
                if (
                    (type(idx) is tuple)
                    and (len(idx) == 2)
                    and (type(idx[0]) is str)
                    and (type(idx[1]) is str)
                ):
                    idx = [idx]
                else:
                    idx = [tuple(i) for i in idx]

        ## If idx is integral, assume it is indexing the structures list,
        ##  otherwise assume it's giving structure name
        if idx_type is int:
            str_idx_list = idx
        else:
            ## Need to find the structures that correspond to this compound(s)
            str_idx_list = [i for c in idx for i in self.compounds[c]]

        str_list = [self.structures[i] for i in str_idx_list]
        compounds = [s["compound"] for s in str_list]
        if return_list:
            return (compounds, str_list)
        else:
            return (compounds[0], str_list[0])

    def __iter__(self):
        for s in self.structures:
            yield (s["compound"], s)
