from asapdiscovery.data.schema import ExperimentalCompoundData
from torch.utils.data import Dataset, Subset


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
        num_workers=1,
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
        num_workers : int, default=1
            Number of cores to use to load structures
        """
        super().__init__()

        # Function to extract from extra_dict (just to make the list
        #  comprehension look a bit nicer)
        get_extra = lambda compound: (  # noqa" E731
            extra_dict[compound] if (extra_dict and (compound in extra_dict)) else None
        )
        mp_args = [
            (fn, compound, ignore_h, lig_resn, get_extra(compound))
            for i, (fn, compound) in enumerate(zip(str_fns, compounds))
        ]

        if num_workers > 1:
            import multiprocessing as mp

            n_procs = min(num_workers, mp.cpu_count(), len(mp_args))
            with mp.Pool(n_procs) as pool:
                all_structures = pool.starmap(DockedDataset._load_structure, mp_args)
        else:
            all_structures = [DockedDataset._load_structure(*args) for args in mp_args]

        self.compounds = {}
        self.structures = []
        for i, (compound, struct) in enumerate(zip(compounds, all_structures)):
            try:
                self.compounds[compound].append(i)
            except KeyError:
                self.compounds[compound] = [i]
            self.structures.append(struct)

    @staticmethod
    def _load_structure(fn, compound, ignore_h=True, lig_resn="LIG", extra_dict=None):
        """
        Helper function to load a single structure that can be multiprocessed in
        the class constructor.

        Parameters
        ----------
        fn : str
            PDB file path
        compound : Tuple[str]
            (crystal structure, ligand compound id)
        ignore_h : bool, default=True
            Whether to remove hydrogens from the loaded structure
        lig_resn : str, default='LIG'
            The residue name for the ligand molecules in the PDB files. Used to
            identify which atoms belong to the ligand
        extra_dict : dict, optional
            Extra information to add to this structure. Values can be anything
            as long as they don't have the keys ["z", "pos", "lig", "compound"]

        Returns
        -------
        """
        import torch
        from Bio.PDB.PDBParser import PDBParser
        from rdkit.Chem import GetPeriodicTable

        table = GetPeriodicTable()

        s = PDBParser().get_structure(f"{compound[0]}_{compound[1]}", fn)
        # Filter out water residues
        all_atoms = [a for a in s.get_atoms() if a.parent.resname != "HOH"]
        if ignore_h:
            all_atoms = [a for a in all_atoms if a.element != "H"]
        # Fix multi-letter atom elements being in all caps (eg CL)
        atomic_nums = [table.GetAtomicNumber(a.element.title()) for a in all_atoms]
        atom_pos = [tuple(a.get_vector()) for a in all_atoms]
        is_lig = [a.parent.resname == lig_resn for a in all_atoms]

        # Create new structure and update from extra_dict
        new_structure = {
            "z": torch.tensor(atomic_nums),
            "pos": torch.tensor(atom_pos).float(),
            "lig": torch.tensor(is_lig),
            "compound": compound,
        }
        if extra_dict:
            new_structure.update(extra_dict)

        return new_structure

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

        # Extract idx from inside the tensor object
        if torch.is_tensor(idx):
            try:
                idx = idx.item()
            except ValueError:
                idx = idx.tolist()

        # Figure out the type of the index, and keep note of whether a list was
        #  passed in or not
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

        # If idx is integral, assume it is indexing the structures list,
        #  otherwise assume it's giving structure name
        if idx_type is int:
            str_idx_list = idx
        else:
            # Need to find the structures that correspond to this compound(s)
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


class GroupedDockedDataset(Dataset):
    """
    Version of DockedDataset where data is grouped by compound_id, so all poses
    for a given compound can be accessed at a time.
    """

    def __init__(
        self,
        str_fns,
        compounds,
        ignore_h=True,
        lig_resn="LIG",
        extra_dict=None,
        num_workers=1,
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
        num_workers : int, default=1
            Number of cores to use to load structures
        """
        import numpy as np

        super().__init__()

        # Function to extract from extra_dict (just to make the list
        #  comprehension look a bit nicer)
        def get_extra(compound):
            return (
                extra_dict[compound]
                if (extra_dict and (compound in extra_dict))
                else None
            )

        mp_args = [
            (fn, compound, ignore_h, lig_resn, get_extra(compound))
            for i, (fn, compound) in enumerate(zip(str_fns, compounds))
        ]

        if num_workers > 1:
            import multiprocessing as mp

            n_procs = min(num_workers, mp.cpu_count(), len(mp_args))
            with mp.Pool(n_procs) as pool:
                all_structures = pool.starmap(DockedDataset._load_structure, mp_args)
        else:
            all_structures = [DockedDataset._load_structure(*args) for args in mp_args]

        self.compound_ids = []
        self.structures = {}
        for i, ((_, compound_id), struct) in enumerate(zip(compounds, all_structures)):
            try:
                self.structures[compound_id].append(struct)
            except KeyError:
                self.structures[compound_id] = [struct]
                self.compound_ids.append(compound_id)
        self.compound_ids = np.asarray(self.compound_ids)

    def __len__(self):
        return len(self.compound_ids)

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int, str, list[str/int], tensor[str/int]
            Index into dataset. Can either be a numerical index into the
            compound ids or the compound id itself, or a
            list/torch.tensor/numpy.ndarray of either of those types.

        Returns
        -------
        List[str]
            List of compound_id for found groups
        List[List[Dict]]
            List of groups (lists) of dict representation of structures
        """
        import torch

        # Extract idx from inside the tensor object
        if torch.is_tensor(idx):
            try:
                idx = idx.item()
            except ValueError:
                idx = idx.tolist()

        # Figure out the type of the index, and keep note of whether a list was
        #  passed in or not
        if (type(idx) is int) or (type(idx) is str):
            return_list = False
            idx_type = type(idx)
            idx = [idx]
        elif (type(idx[0]) is int) or (type(idx[0]) is str):
            return_list = True
            idx_type = type(idx[0])
        else:
            try:
                err_type = type(idx[0])
            except TypeError:
                err_type = type(idx)
            raise TypeError(f"Unknown indexing type {err_type}")

        # If idx is integral, assume it is indexing the structures list,
        #  otherwise assume it's giving structure name
        if idx_type is int:
            compound_id_list = self.compound_ids[idx]
        else:
            compound_id_list = idx

        str_list = [self.structures[compound_id] for compound_id in compound_id_list]
        if return_list:
            return compound_id_list, str_list
        else:
            return compound_id_list[0], str_list[0]

    def __iter__(self):
        for compound_id in self.compound_ids:
            yield compound_id, self.structures[compound_id]


class GraphDataset(Dataset):
    """
    Class for loading SMILES as graphs.
    """

    def __init__(
        self,
        exp_compounds,
        node_featurizer=None,
        edge_featurizer=None,
        cache_file=None,
    ):
        """
        Parameters
        ----------
        exp_compounds : List[schema.ExperimentalCompoundData]
            List of compounds
        node_featurizer : BaseAtomFeaturizer, optional
            Featurizer for node data
        edge_featurizer : BaseBondFeaturizer, optional
            Featurizer for edges
        cache_file : str, optional
            Cache file for graph dataset

        """
        import pandas
        from dgllife.data import MoleculeCSVDataset
        from dgllife.utils import SMILESToBigraph

        # Build dataframe
        all_compound_ids, all_smiles, all_pic50, all_range, all_stderr, all_dates = zip(
            *[
                (
                    c.compound_id,
                    c.smiles,
                    c.experimental_data["pIC50"],
                    c.experimental_data["pIC50_range"],
                    c.experimental_data["pIC50_stderr"],
                    c.date_created,
                )
                for c in exp_compounds
            ]
        )
        df = pandas.DataFrame(
            {
                "compound_id": all_compound_ids,
                "smiles": all_smiles,
                "pIC50": all_pic50,
                "pIC50_range": all_range,
                "pIC50_stderr": all_stderr,
            }
        )

        # Build dataset
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
            task_names=["pIC50", "pIC50_range", "pIC50_stderr"],
        )

        # Build dict mapping compound to date
        dates_dict = dict(zip(all_compound_ids, all_dates))

        self.compounds = {}
        self.structures = []
        for i, (compound_id, g) in enumerate(zip(all_compound_ids, dataset)):
            # Need a tuple to match DockedDataset, but the graph objects aren't
            #  attached to a protein structure at all
            compound = ("NA", compound_id)

            # Add data
            try:
                self.compounds[compound].append(i)
            except KeyError:
                self.compounds[compound] = [i]
            self.structures.append(
                {
                    "smiles": g[0],
                    "g": g[1],
                    "pIC50": g[2][0],
                    "pIC50_range": g[2][1],
                    "pIC50_stderr": g[2][2],
                    "date_created": dates_dict[compound_id],
                    "compound": compound,
                }
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

        # Extract idx from inside the tensor object
        if torch.is_tensor(idx):
            try:
                idx = idx.item()
            except ValueError:
                idx = idx.tolist()

        # Figure out the type of the index, and keep note of whether a list was
        #  passed in or not
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

        # If idx is integral, assume it is indexing the structures list,
        #  otherwise assume it's giving structure name
        if idx_type is int:
            str_idx_list = idx
        else:
            # Need to find the structures that correspond to this compound(s)
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


class GraphInferenceDataset(Dataset):
    """
    Class for loading SMILES as graphs without experimental data
    """

    def __init__(
        self,
        exp_compounds,
        node_featurizer=None,
        edge_featurizer=None,
        cache_file="./cache.bin",
    ):
        """
        Parameters
        ----------
        exp_compounds : Union[List[schema.ExperimentalCompoundData], List[str]]
            List of compounds or smiles
        node_featurizer : BaseAtomFeaturizer, optional
            Featurizer for node data
        edge_featurizer : BaseBondFeaturizer, optional
            Featurizer for edges
        cache_file : str, optional
            Cache file for graph dataset

        """
        import pandas
        from dgllife.data import MoleculeCSVDataset
        from dgllife.utils import SMILESToBigraph

        self.compounds_dict = {}
        self.smiles_dict = {}
        self.graphs = []

        if all([type(exp) == str for exp in exp_compounds]):
            exp_compounds = [
                ExperimentalCompoundData(compound_id=i, smiles=c)
                for i, c in enumerate(exp_compounds)
            ]
        elif all([type(exp) == ExperimentalCompoundData for exp in exp_compounds]):
            pass
        else:
            raise TypeError(
                "exp_compounds must be a list of strings or ExperimentalCompoundData"
            )

        # Build dataframe
        all_compound_ids, all_smiles = zip(
            *[
                (
                    c.compound_id,
                    c.smiles,
                )
                for c in exp_compounds
            ]
        )
        df = pandas.DataFrame(
            {
                "compound_id": all_compound_ids,
                "smiles": all_smiles,
            }
        )

        # Build dataset
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
            task_names=[],
        )

        for compound_id, g in zip(all_compound_ids, dataset):
            self.compounds_dict[compound_id] = g[1]
            self.graphs.append(g[1])
            self.smiles_dict[g[0]] = g[1]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.smiles_dict[idx]
        elif isinstance(idx, int):
            return self.graphs[idx]
        elif isinstance(idx, list):
            return [self.graphs[i] for i in idx]
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            idx = list(range(start, stop, step))
            subset = Subset(self, idx)
            return subset
        else:
            raise TypeError("idx must be a string, int, list, or slice")

    def __iter__(self):
        yield from self.graphs
