from openeye import oechem
import numpy as np

def load_openeye_pdb(pdb_fn):
    ifs = oechem.oemolistream()
    ifs.SetFlavor(
        oechem.OEFormat_PDB,
        oechem.OEIFlavor_PDB_Default | oechem.OEIFlavor_PDB_DATA,
    )
    ifs.open(pdb_fn)
    in_mol = oechem.OEGraphMol()
    oechem.OEReadMolecule(ifs, in_mol)
    ifs.close()

    return in_mol

def load_openeye_sdf(sdf_fn):
    ifs = oechem.oemolistream()
    ifs.SetFlavor(
        oechem.OEFormat_SDF,
        oechem.OEIFlavor_SDF_Default,
    )
    ifs.open(sdf_fn)
    coords_mol = oechem.OEGraphMol()
    oechem.OEReadMolecule(ifs, coords_mol)
    ifs.close()

    return coords_mol

def split_openeye_mol(complex_mol: oechem.OEMolBase):
    ## Test splitting
    lig_mol = oechem.OEGraphMol()
    prot_mol = oechem.OEGraphMol()
    water_mol = oechem.OEGraphMol()
    oth_mol = oechem.OEGraphMol()

    ## Make splitting split out covalent ligands
    ## TODO: look into different covalent-related options here
    opts = oechem.OESplitMolComplexOptions()
    opts.SetSplitCovalent(True)
    opts.SetSplitCovalentCofactors(True)
    print(
        oechem.OESplitMolComplex(lig_mol, prot_mol, water_mol, oth_mol, complex_mol)
    )

    print(
        complex_mol.NumAtoms(),
        lig_mol.NumAtoms(),
        prot_mol.NumAtoms(),
        water_mol.NumAtoms(),
        oth_mol.NumAtoms(),
    )
    return {'complex': complex_mol,
            'lig': lig_mol,
            'pro': prot_mol,
            'water': water_mol,
            'other': oth_mol}