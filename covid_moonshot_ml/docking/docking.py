import os

from kinoml.core.proteins import Protein
from kinoml.core.ligands import Ligand
from kinoml.core.systems import ProteinLigandComplex

def build_docking_systems(
    exp_compounds, xtal_compounds, compound_idxs, n_top=1
):
    """
    Build systems to run through docking.
    Parameters
    ----------
    exp_compounds : list[schema.ExperimentalCompoundData]
        List of compounds to dock
    xtal_compounds : list[schema.CrystalCompoundData]
        List of all crystal structures
    compound_idxs : list[int]
        List giving the index of the crystal structure to dock to for each
        ligand. Should be the same length as `exp_compounds`
    n_top : int, default=1
        Dock to top `n_top` crystal structures

    Returns
    -------
    List[kinoml.core.systems.ProteinLigandComplex]
        List of protein+ligand systems for docking
    """
    systems = []
    for (c, idx) in zip(exp_compounds, compound_idxs):
        ## Make sure that there are enough crystal structures to dock to
        n_dock = min(n_top, len(idx))
        for i in range(n_dock):
            ## Build protein, ligand, and complex objects
            x = xtal_compounds[idx[i]]
            protein = Protein.from_file(x.str_fn, name="MPRO")
            protein.chain_id = x.str_fn.split("_")[-2][-1]
            protein.expo_id = "LIG"
            ligand = Ligand.from_smiles(smiles=c.smiles, name=c.compound_id)
            systems.append(ProteinLigandComplex(components=[protein, ligand]))

    return systems


def build_docking_system_direct(prot_mol, lig_smi, prot_name, lig_name):
    """
    Build system to run through kinoml docking from OEGraphMol objects.

    Parameters
    ----------
    prot_mol : oechem.OEGraphMol
        Protein molecule.
    lig_smi : str
        Ligand SMILES string.
    prot_name : str
        Name of protein.
    lig_name : str
        Name of ligand.

    Returns
    -------
    kinoml.core.systems.ProteinLigandComplex
    """
    protein = Protein(molecule=prot_mol, name=prot_name)
    ligand = Ligand.from_smiles(smiles=lig_smi, name=lig_name)

    return ProteinLigandComplex(components=[protein, ligand])


def build_combined_protein_system_from_sdf(pdb_fn, sdf_fn):
    protein = Protein.from_file(pdb_fn, name="MERS-Mpro")
    ligand = Ligand.from_file(sdf_fn)
    return ProteinLigandComplex


def parse_xtal(x_fn, x_dir):
    """
    Load all crystal structures into schema.CrystalCompoundData objects.
    Parameters
    ----------
    x_fn : str
        CSV file giving information on each crystal structure
    x_dir : str
        Path to directory containing directories with crystal structure PDB
        files
    Returns
    -------
    List[schema.CrystalCompoundData]
        List of parsed crystal structures
    """
    import pandas

    from ..schema import CrystalCompoundData

    df = pandas.read_csv(x_fn)

    ## Find all P-files
    idx = [(type(d) is str) and ("-P" in d) for d in df["Dataset"]]

    ## Build argument dicts for the CrystalCompoundData objects
    xtal_dicts = [
        dict(zip(("smiles", "dataset"), r[1].values))
        for r in df.loc[idx, ["SMILES", "Dataset"]].iterrows()
    ]

    ## Add structure filename information
    for d in xtal_dicts:
        fn_base = (
            f'{x_dir}/{d["dataset"]}_0{{}}/{d["dataset"]}_0{{}}_' "seqres.pdb"
        )
        fn = fn_base.format("A", "A")
        if os.path.isfile(fn):
            d["str_fn"] = fn
        else:
            fn = fn_base.format("B", "B")
            assert os.path.isfile(fn), f'No structure found for {d["dataset"]}.'
            d["str_fn"] = fn

    ## Build CrystalCompoundData objects for each row
    xtal_compounds = [CrystalCompoundData(**d) for d in xtal_dicts]

    return xtal_compounds


def run_docking(cache_dir, output_dir, loop_db, n_procs, docking_systems):
    from kinoml.features.complexes import OEDockingFeaturizer

    featurizer = OEDockingFeaturizer(
        cache_dir=cache_dir,
        output_dir=output_dir,
        loop_db=loop_db,
        n_processes=n_procs,
    )
    docking_systems = featurizer.featurize(docking_systems)

    return docking_systems
