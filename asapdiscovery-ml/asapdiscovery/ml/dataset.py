import numpy as np
import pandas as pd
import torch
from asapdiscovery.data.backend.openeye import oechem
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.schema.ligand import Ligand
from torch.utils.data import Dataset


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
    def from_complexes(cls, complexes: list[Complex], exp_dict=None, ignore_h=True):
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

        if exp_dict is None:
            exp_dict = {}

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
        # Can't use enumerate in case we skip some
        comp_counter = 0
        for comp in complexes:
            try:
                comp_exp_dict = comp.ligand.experimental_data.experimental_data
            except AttributeError:
                comp_exp_dict = {}
            comp_exp_dict |= exp_dict.get(comp.ligand.compound_name, {})

            # compound = get_complex_id(comp)
            compound = (comp.target.target_name, comp.ligand.compound_name)
            try:
                compound_idxs[compound].append(comp_counter)
            except KeyError:
                compound_idxs[compound] = [comp_counter]

            pose = cls._complex_to_pose(
                comp, compound=compound, exp_dict=comp_exp_dict, ignore_h=ignore_h
            )
            structures.append(pose)

            comp_counter += 1

        return cls(compound_idxs, structures)

    @staticmethod
    def _complex_to_pose(comp, compound=None, exp_dict=None, ignore_h=True):
        """
        Helper function to convert a Complex to a pose.
        """

        if exp_dict is None:
            exp_dict = {}

        # First get target atom positions, atomic numbers, and B factors
        target_mol = comp.target.to_oemol()
        target_coords = target_mol.GetCoords()
        target_pos = []
        target_z = []
        target_b = []
        for atom in target_mol.GetAtoms():
            target_pos.append(target_coords[atom.GetIdx()])
            target_z.append(atom.GetAtomicNum())
            target_b.append(oechem.OEAtomGetResidue(atom).GetBFactor())

        # Get ligand atom positions, atomic numbers, and B factors
        ligand_mol = comp.ligand.to_oemol()
        ligand_coords = ligand_mol.GetCoords()
        ligand_pos = []
        ligand_z = []
        ligand_b = []
        for atom in ligand_mol.GetAtoms():
            ligand_pos.append(ligand_coords[atom.GetIdx()])
            ligand_z.append(atom.GetAtomicNum())
            ligand_b.append(oechem.OEAtomGetResidue(atom).GetBFactor())

        # Combine the two
        all_pos = torch.tensor(target_pos + ligand_pos).float()
        all_z = torch.tensor(target_z + ligand_z)
        all_b = torch.tensor(target_b + ligand_b)
        all_lig = torch.tensor(
            [False] * target_mol.NumAtoms() + [True] * ligand_mol.NumAtoms()
        )

        # Add some extra stuff for use in e3nn models
        all_one_hot = torch.nn.functional.one_hot(all_z - 1, 100).float()

        # Subset to remove Hs if desired
        if ignore_h:
            h_idx = all_z == 1
            all_pos = all_pos[~h_idx]
            all_z = all_z[~h_idx]
            all_b = all_b[~h_idx]
            all_lig = all_lig[~h_idx]
            all_one_hot = all_one_hot[~h_idx]

        pose = {
            "pos": all_pos,
            "z": all_z,
            "lig": all_lig,
            "x": all_one_hot,
            "b": all_b,
            "ligand": comp.ligand,
        }
        if compound:
            pose["compound"] = compound
        return pose | exp_dict

    @classmethod
    def from_files(
        cls,
        str_fns,
        compounds,
        ignore_h=True,
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
        extra_dict : dict[str, dict], optional
            Extra information to add to each structure. Keys should be
            compounds, and dicts can be anything as long as they don't have the
            keys ["z", "pos", "lig", "compound"]
        num_workers : int, default=1
            Number of cores to use to load structures
        """
        if extra_dict is None:
            extra_dict = {}

        mp_args = [(fn, compound) for fn, compound in zip(str_fns, compounds)]

        def mp_func(fn, compound):
            return Complex.from_pdb(
                pdb_file=fn,
                target_kwargs={"target_name": compound[0]},
                ligand_kwargs={"compound_name": compound[1]},
            )

        if num_workers > 1:
            import multiprocessing as mp

            n_procs = min(num_workers, mp.cpu_count(), len(mp_args))
            with mp.Pool(n_procs) as pool:
                all_complexes = pool.starmap(mp_func, mp_args)
        else:
            all_complexes = [mp_func(*args) for args in mp_args]

        return cls.from_complexes(all_complexes, exp_dict=extra_dict, ignore_h=ignore_h)

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
        elif isinstance(idx, slice):
            return_list = True
            idx_type = int
            start, stop, step = idx.indices(len(self))
            idx = list(range(start, stop, step))
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
            return list(zip(compounds, str_list))
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
        from asapdiscovery.docking.analysis import calculate_rmsd_openeye

        compound_ids = []
        structures = {}
        for i, comp in enumerate(complexes):
            # compound = get_complex_id(comp)
            compound = (comp.target.target_name, comp.ligand.compound_name)

            # Build pose dict
            try:
                comp_exp_dict = comp.ligand.experimental_data.experimental_data
            except AttributeError:
                comp_exp_dict = {}
            comp_exp_dict |= exp_dict.get(comp.ligand.compound_name, {})
            pose = DockedDataset._complex_to_pose(
                comp, compound=compound, exp_dict=comp_exp_dict, ignore_h=ignore_h
            )

            # Calculate RMSD to ref if available
            if "xtal_ligand" in pose:
                pose["ref_rmsd"] = calculate_rmsd_openeye(
                    Ligand(**pose["xtal_ligand"]).to_oemol(), pose["ligand"].to_oemol()
                )

            try:
                structures[comp.ligand.compound_name]["poses"].append(pose)
            except KeyError:
                # Take compound-level data from first pose
                exp_data = {
                    k: v
                    for k, v in pose.items()
                    if (not isinstance(v, torch.Tensor)) and (k != "ref_rmsd")
                }
                structures[comp.ligand.compound_name] = {"poses": [pose]} | exp_data
                compound_ids.append(comp.ligand.compound_name)

        # Calculate which pose is closest to experiment
        for compound_id, data in structures.items():
            if "xtal_ligand" not in data:
                continue

            # Get all RMSDs
            pose_rmsds = np.asarray([pose["ref_rmsd"] for pose in data["poses"]])

            # Label of all zeros, except the one with the best pose (lowest ref RMSD)
            best_lab = np.zeros(len(data["poses"]))
            best_lab[np.argmin(pose_rmsds)] = 1

            data["best_pose_label"] = best_lab
            # Normalize to probability, take inverse first so lower RMSDs are better
            data["rmsd_probs"] = (1 / pose_rmsds) / (1 / pose_rmsds).sum()

        return cls(compound_ids=compound_ids, structures=structures)

    @classmethod
    def from_files(
        cls,
        str_fns,
        compounds,
        ignore_h=True,
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
        extra_dict : dict[str, dict], optional
            Extra information to add to each structure. Keys should be
            compounds, and dicts can be anything as long as they don't have the
            keys ["z", "pos", "lig", "compound"]
        num_workers : int, default=1
            Number of cores to use to load structures
        """

        if extra_dict is None:
            extra_dict = {}

        mp_args = [(fn, compound) for fn, compound in zip(str_fns, compounds)]

        def mp_func(fn, compound):
            return Complex.from_pdb(
                pdb_file=fn,
                target_kwargs={"target_name": compound[0]},
                ligand_kwargs={"compound_name": compound[1]},
            )

        if num_workers > 1:
            import multiprocessing as mp

            n_procs = min(num_workers, mp.cpu_count(), len(mp_args))
            with mp.Pool(n_procs) as pool:
                all_complexes = pool.starmap(mp_func, mp_args)
        else:
            all_complexes = [mp_func(*args) for args in mp_args]

        return cls.from_complexes(all_complexes, exp_dict=extra_dict, ignore_h=ignore_h)

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
        elif isinstance(idx, slice):
            return_list = True
            idx_type = int
            start, stop, step = idx.indices(len(self))
            idx = list(range(start, stop, step))
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
            return list(zip(compound_id_list, str_list))
        else:
            return (compound_id_list[0], str_list[0])

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
        """

        from dgllife.utils import SMILESToBigraph

        # Function for encoding SMILES to a graph
        smiles_to_g = SMILESToBigraph(
            add_self_loop=True,
            node_featurizer=node_featurizer,
            edge_featurizer=edge_featurizer,
        )

        compounds = {}
        structures = []
        for i, lig in enumerate(ligands):
            compound_id = lig.compound_name
            smiles = lig.smiles

            # Need a tuple to match DockedDataset, but the graph objects aren't
            #  attached to a protein structure at all
            compound = ("NA", compound_id)

            # Generate DGL graph
            g = smiles_to_g(smiles)

            # Gather experimental data
            try:
                lig_exp_dict = lig.experimental_data.experimental_data
                if lig.experimental_data.date_created:
                    lig_exp_dict |= {"date_created": lig.experimental_data.date_created}
            except AttributeError:
                lig_exp_dict = {}
            lig_exp_dict |= exp_dict.get(compound_id, {})

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
                | lig_exp_dict
            )

        return cls(compounds, structures)

    @classmethod
    def from_exp_compounds(
        cls,
        exp_compounds,
        exp_dict: dict = {},
        node_featurizer=None,
        edge_featurizer=None,
    ):
        """
        Parameters
        ----------
        exp_compounds : List[schema.ExperimentalCompoundData]
            List of compounds
        exp_dict : dict[str, dict[str, int | float]], optional
            Dict mapping compound_id to an experimental results dict. The dict for a
            compound will be added to the pose representation of each Complex containing
            a ligand witht that compound_id
        node_featurizer : BaseAtomFeaturizer, optional
            Featurizer for node data
        edge_featurizer : BaseBondFeaturizer, optional
            Featurizer for edges
            Cache file for graph dataset

        """
        from dgllife.utils import SMILESToBigraph

        # Function for encoding SMILES to a graph
        smiles_to_g = SMILESToBigraph(
            add_self_loop=True,
            node_featurizer=node_featurizer,
            edge_featurizer=edge_featurizer,
        )

        compounds = {}
        structures = []
        for i, exp_compound in enumerate(exp_compounds):
            compound_id = exp_compound.compound_id
            smiles = exp_compound.smiles

            # Need a tuple to match DockedDataset, but the graph objects aren't
            #  attached to a protein structure at all
            compound = ("NA", compound_id)

            # Generate DGL graph
            g = smiles_to_g(smiles)

            # Gather experimental data
            lig_exp_dict = exp_compound.experimental_data.copy()
            lig_exp_dict |= exp_dict.get(compound_id, {})
            if exp_compound.date_created:
                lig_exp_dict |= {"date_created": exp_compound.date_created}

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
                | exp_compound.experimental_data
                | exp_dict.get(compound_id, {})
                | {"date_created": exp_compound.date_created}
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
        elif isinstance(idx, slice):
            return_list = True
            idx_type = int
            start, stop, step = idx.indices(len(self))
            idx = list(range(start, stop, step))
        else:
            return_list = True
            if isinstance(idx[0], bool):
                idx_type = bool
                if len(idx) != len(self.structures):
                    raise IndexError("Index length must match number of structures.")
            elif isinstance(idx[0], int):
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
        elif idx_type is bool:
            str_idx_list = [i for i in range(len(self.structures)) if idx[i]]
        else:
            # Need to find the structures that correspond to this compound(s)
            str_idx_list = [i for c in idx for i in self.compounds[c]]

        str_list = [self.structures[i] for i in str_idx_list]
        compounds = [s["compound"] for s in str_list]
        if return_list:
            return list(zip(compounds, str_list))
        else:
            return (compounds[0], str_list[0])

    def __iter__(self):
        for s in self.structures:
            yield (s["compound"], s)


def dataset_to_dataframe(dataset):
    all_data = []
    for k, v in dataset:
        # add all string castable data in v to a dict
        data_dict = {}
        for key, value in v.items():
            try:
                value = str(value)
                data_dict[key] = value
            except:  # noqa: E722
                pass

        # add compound tuple to dict
        data_dict["xtal_id"] = k[0]
        data_dict["compound_id"] = k[1]
        all_data.append(data_dict)
    return pd.DataFrame(all_data)


def dataset_to_csv(dataset, filename):
    dataset_to_dataframe(dataset).to_csv(filename, index=False)
    return filename
