import torch
from asapdiscovery.data.schema import ExperimentalCompoundData
from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.ligand import Ligand
from torch.utils.data import Dataset, Subset


class DockedDataset(Dataset):
    """
    Class for loading docking results into a dataset to be used for graph
    learning.
    """

    def __init__(self, compounds={}, structures=[]):
        """
        Constructor for DockedDataset object.

        Parameters
        ----------
        compounds : dict[(str, str), list[int]]
            Dict mapping a compound tuple (xtal_id, compound_id) to a list of indices in
            structures that are poses for that id pair
        structures : list[dict]
            List of pose dicts, containing at minimum tensors for atomic number, atomic
            positions, and a ligand idx. Indices in this list should match the indices
            in the lists in compounds.
        """
        super().__init__()

        self.compounds = compounds
        self.structures = structures

    @classmethod
    def from_complexes(cls, complexes: list[Complex], exp_dict={}, ignore_h=True):
        """
        Build from a list of Complex objects.

        Parameters
        ----------
        complexes : list[Complex]
            List of Complex schema objects to build into a DockedDataset object
        exp_dict : dict[str, dict[str, int | float]], optional
            Dict mapping compound_id to an experimental results dict. The dict for a
            compound will be added to the pose representation of each Complex containing
            a ligand witht that compound_id
        ignore_h : bool, default=True
            Whether to remove hydrogens from the loaded structure

        Returns
        -------
        DockedDataset
        """

        # Helper function to grab all relevant
        def get_complex_id(c):
            # First build target id from target_name and all identifiers
            target_name = c.target.target_name
            target_ids = {k: v for k, v in c.target.ids.dict() if v}
            target_id = []
            if target_name:
                target_id += [target_name]
            if len(target_ids):
                target_id += [target_ids]

            # Build ligand_id from compound_name and all identifiers
            compound_name = c.ligand.compound_name
            compound_ids = {k: v for k, v in c.ligand.ids.dict() if v}
            compound_id = []
            if compound_name:
                compound_id += [compound_name]
            if len(compound_ids):
                compound_id += [compound_ids]

            return tuple(target_id), tuple(compound_id)

        compound_idxs = {}
        structures = []
        for i, comp in enumerate(complexes):
            # compound = get_complex_id(comp)
            compound = (comp.target.target_name, comp.ligand.compound_name)
            try:
                compound_idxs[compound].append(i)
            except KeyError:
                compound_idxs[compound] = [i]

            comp_exp_dict = exp_dict.get(comp.ligand.compound_name, {})
            pose = cls._complex_to_pose(
                comp, compound=compound, exp_dict=comp_exp_dict, ignore_h=ignore_h
            )
            structures.append(pose)

        return cls(compound_idxs, structures)

    @staticmethod
    def _complex_to_pose(comp, compound=None, exp_dict={}, ignore_h=True):
        """
        Helper function to convert a Complex to a pose.
        """

        # First get target atom positions and atomic numbers
        target_mol = comp.target.to_oemol()
        target_coords = target_mol.GetCoords()
        target_pos = []
        target_z = []
        for atom in target_mol.GetAtoms():
            target_pos.append(target_coords[atom.GetIdx()])
            target_z.append(atom.GetAtomicNum())

        # Get ligand atom positions and atomic numbers
        ligand_mol = comp.ligand.to_oemol()
        ligand_coords = ligand_mol.GetCoords()
        ligand_pos = []
        ligand_z = []
        for atom in ligand_mol.GetAtoms():
            ligand_pos.append(ligand_coords[atom.GetIdx()])
            ligand_z.append(atom.GetAtomicNum())

        # Combine the two
        all_pos = torch.tensor(target_pos + ligand_pos).float()
        all_z = torch.tensor(target_z + ligand_z)
        all_lig = torch.tensor(
            [False] * target_mol.NumAtoms() + [True] * ligand_mol.NumAtoms()
        )

        # Subset to remove Hs if desired
        if ignore_h:
            h_idx = all_z == 1
            all_pos = all_pos[~h_idx]
            all_z = all_z[~h_idx]
            all_lig = all_lig[~h_idx]

        pose = {"pos": all_pos, "z": all_z, "lig": all_lig}
        if compound:
            pose["compound"] = compound
        return pose | exp_dict

    @classmethod
    def from_files(
        cls,
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

        # Function to extract from extra_dict (just to make the list
        #  comprehension look a bit nicer)
        def get_extra(compound):
            return (
                extra_dict[compound] if extra_dict and compound in extra_dict else None
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

        compound_idxs = {}
        structures = []
        for i, (compound, struct) in enumerate(zip(compounds, all_structures)):
            try:
                compound_idxs[compound].append(i)
            except KeyError:
                compound_idxs[compound] = [i]
            structures.append(struct)

        return cls(compound_idxs, structures)

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

        if compound is None:
            # use the stem of the name of the file as the PDB id
            id = str(fn).split("/")[-1].split(".")[0]
        else:
            # use the compound info as the PDB id
            id = f"{compound[0]}_{compound[1]}"
        s = PDBParser().get_structure(id, fn)
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
        if isinstance(idx, int):
            return_list = False
            idx_type = int
            idx = [idx]
        else:
            return_list = True
            if isinstance(idx[0], int):
                idx_type = int
            else:
                idx_type = tuple
                if (
                    isinstance(idx, tuple)
                    and (len(idx) == 2)
                    and isinstance(idx[0], str)
                    and isinstance(idx[1], str)
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

    def __init__(self, compound_ids: list[str] = [], structures: dict[str, dict] = {}):
        """
        Constructor for GroupedDockedDataset object.

        Parameters
        ----------
        compound_ids : list[str]
            List of compound ids. Each entry in this list must have a corresponding
            entry in structures
        structures : dict[str, dict]
            Dict mapping compound_id to a pose dict
        """
        import numpy as np

        super().__init__()

        self.compound_ids = np.asarray(compound_ids)
        self.structures = structures

    @classmethod
    def from_complexes(cls, complexes: list[Complex], exp_dict={}, ignore_h=True):
        """
        Build from a list of Complex objects.

        Parameters
        ----------
        complexes : list[Complex]
            List of Complex schema objects to build into a DockedDataset object
        exp_dict : dict[str, dict[str, int | float]], optional
            Dict mapping compound_id to an experimental results dict. The dict for a
            compound will be added to the pose representation of each Complex containing
            a ligand witht that compound_id
        ignore_h : bool, default=True
            Whether to remove hydrogens from the loaded structure

        Returns
        -------
        GroupedDockedDataset
        """

        compound_ids = []
        structures = {}
        for i, comp in enumerate(complexes):
            # compound = get_complex_id(comp)
            compound = (comp.target.target_name, comp.ligand.compound_name)

            # Build pose dict
            comp_exp_dict = exp_dict.get(comp.ligand.compound_name, {})
            pose = DockedDataset._complex_to_pose(
                comp, compound=compound, exp_dict=comp_exp_dict, ignore_h=ignore_h
            )
            try:
                structures[comp.ligand.compound_name].append(pose)
            except KeyError:
                structures[comp.ligand.compound_name] = [pose]
                compound_ids.append(comp.ligand.compound_name)

        return cls(compound_ids=compound_ids, structures=structures)

    @classmethod
    def from_files(
        cls,
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

        compound_ids = []
        structures = {}
        for i, ((_, compound_id), struct) in enumerate(zip(compounds, all_structures)):
            try:
                structures[compound_id].append(struct)
            except KeyError:
                structures[compound_id] = [struct]
                compound_ids.append(compound_id)
        compound_ids = np.asarray(compound_ids)

        return cls(compound_ids=compound_ids, structures=structures)

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
        if (isinstance(idx, int)) or (isinstance(idx, str)):
            return_list = False
            idx_type = type(idx)
            idx = [idx]
        elif (isinstance(idx[0], int)) or (isinstance(idx[0], str)):
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

    def __init__(self, compounds={}, structures=[]):
        super().__init__()

        self.compounds = compounds
        self.structures = structures

    @classmethod
    def from_ligands(
        cls,
        ligands: list[Ligand],
        exp_dict: dict = {},
        node_featurizer=None,
        edge_featurizer=None,
        cache_file=None,
    ):
        """
        Parameters
        ----------
        ligands : list[Ligands]
            List of Ligand schema objects to build into a GraphDataset object
        exp_dict : dict[str, dict[str, int | float]], optional
            Dict mapping compound_id to an experimental results dict. The dict for a
            compound will be added to the pose representation of each Complex containing
            a ligand witht that compound_id
        node_featurizer : BaseAtomFeaturizer, optional
            Featurizer for node data
        edge_featurizer : BaseBondFeaturizer, optional
            Featurizer for edges
        cache_file : str, optional
            Cache file for graph dataset
        """
        from functools import reduce

        import numpy as np
        from dgllife.utils import SMILESToBigraph

        # Make sure that all exp dicts for each compound have all the same keys so the
        #  DF can be properly constructed
        all_shared_exp_keys = [set(d.keys()) for d in exp_dict.values()]
        if len(all_shared_exp_keys) > 0:
            all_shared_exp_keys = reduce(
                lambda s1, s2: s1.intersection(s2), all_shared_exp_keys
            )
        exp_dict = {
            compound_id: {k: v for k, v in d.items() if k in all_shared_exp_keys}
            for compound_id, d in exp_dict.items()
        }

        # Organize data for DF construction
        missing_dict = {k: np.nan for k in all_shared_exp_keys}
        all_info = []
        for lig in ligands:
            lig_exp_dict = exp_dict.get(lig.compound_name, missing_dict)
            exp_info = [lig_exp_dict[k] for k in all_shared_exp_keys]
            all_info.append([lig.compound_name, lig.smiles] + exp_info)

        # Function for encoding SMILES to a graph
        smiles_to_g = SMILESToBigraph(
            add_self_loop=True,
            node_featurizer=node_featurizer,
            edge_featurizer=edge_featurizer,
        )

        compounds = {}
        structures = []
        for i, compound_info in enumerate(all_info):
            compound_id, smiles, *exp_info = compound_info
            # Need a tuple to match DockedDataset, but the graph objects aren't
            #  attached to a protein structure at all
            compound = ("NA", compound_id)

            # Generate DGL graph
            g = smiles_to_g(smiles)

            # Add data
            try:
                compounds[compound].append(i)
            except KeyError:
                compounds[compound] = [i]
            structures.append(
                {
                    "smiles": smiles,
                    "g": g,
                    "compound": compound,
                }
                | dict(zip(list(all_shared_exp_keys), exp_info))
            )

        return cls(compounds, structures)

    @classmethod
    def from_exp_compounds(
        cls,
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

        compounds = {}
        structures = []
        for i, (compound_id, g) in enumerate(zip(all_compound_ids, dataset)):
            # Need a tuple to match DockedDataset, but the graph objects aren't
            #  attached to a protein structure at all
            compound = ("NA", compound_id)

            # Add data
            try:
                compounds[compound].append(i)
            except KeyError:
                compounds[compound] = [i]
            structures.append(
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

        return cls(compounds, structures)

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
        if isinstance(idx, int):
            return_list = False
            idx_type = int
            idx = [idx]
        else:
            return_list = True
            if isinstance(idx[0], int):
                idx_type = int
            else:
                idx_type = tuple
                if (
                    isinstance(idx, tuple)
                    and (len(idx) == 2)
                    and isinstance(idx[0], str)
                    and isinstance(idx[1], str)
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

        if all([isinstance(exp, str) for exp in exp_compounds]):
            exp_compounds = [
                ExperimentalCompoundData(compound_id=i, smiles=c)
                for i, c in enumerate(exp_compounds)
            ]
        elif all([type(exp) is ExperimentalCompoundData for exp in exp_compounds]):
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
