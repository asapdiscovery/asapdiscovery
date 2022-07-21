import os.path

import yaml
from kinoml.databases.pdb import download_pdb_structure
# from kinoml.modeling.OEModeling import superpose_proteins
from openeye import oechem
from typing import Iterable
from MDAnalysis.coordinates.PDB import PDBReader, PDBWriter
# from MDAnalysis.analysis.align import alignto
# import mdtraj as md
# from mdtraj.formats import PDBTrajectoryFile





def load_pdbs_from_yaml(pdb_list_yaml):
    print(f"Loading pdb list from {pdb_list_yaml}...")
    with open(pdb_list_yaml, 'r') as f:
        pdb_list = yaml.safe_load(f)
    return pdb_list


def download_PDBs(pdb_list, pdb_path):
    """
    Downloads pdbs from pdb_list_yaml using Kinoml.

    Parameters
    ----------
    pdb_list_yaml
    pdb_path

    Returns
    -------

    """
    print(f"Downloading PDBs to {pdb_path}")
    for pdb in pdb_list:
        print(pdb)
        download_pdb_structure(pdb, pdb_path)


# def align_pdb_to_reference(pdb_path, ref_path):
#     ## first load in pdb files using mdanalysis
#
#     pdb = md.load_pdb(pdb_path)
#     ref = md.load_pdb(ref_path)
#     pdb.superpose(ref, atom_indices=pdb.topology.select("chainid 0 and protein and backbone"))
#     pdb_list = os.path.split(pdb_path)
#     ref_name = os.path.basename(ref_path)
#     new_pdb_path = os.path.join(pdb_list[0], f"aligned_to_{ref_name}_{pdb_list[1]}")
#     # new_pdb = md.formats.PDBTrajectoryFile(new_pdb_path, mode='w')
#
#     pdb.save(new_pdb_path)

# new_pdb.write(
#     pdb.xyz,
#     pdb.topology,
# )
# pdb = PDBTrajectoryFile(pdb_path)
# ref = PDBTrajectoryFile(ref_path)
#
# pdb.superpose(ref, atom_indices=pdb.topology.select("chainid 0 and backbone"))


# (pdb, ref, select="protein and name CA and chain A", weights="mass")]


def mdanalysis_alignment(pdb_path, ref_path, out_path):
    import MDAnalysis as mda
    from MDAnalysis.analysis import align
    from MDAnalysis.analysis.rms import rmsd
    from MDAnalysis.coordinates import PDB

    mobile = mda.Universe(pdb_path, pdb_path)
    ref = mda.Universe(ref_path, ref_path)

    print(len(mobile.select_atoms("protein and segid A and name CA and resnum 147 to 160")))

    print(len(ref.select_atoms("protein and segid A and name CA and resnum 147 to 160")))
    # for segment in mobile.segments:
    #     print(segment.ix)
    #     print(segment.residues)
    #
    # for segment in ref.segments:
    #     print(segment.ix)
    #     print(segment.residues)

    # _ref = md.load_pdb(ref_path)
    # mobile = mda.Universe(pdb)
    # ref = mda.Universe(_ref)
    align.alignto(mobile, ref,
                  select="protein and segid A and name CA", # and resnum 140 to 160",
                  weights="mass",
                  # match_atoms = False,
                  strict=False
                  )
    mobile.atoms.write(out_path)
    return


def pymol_alignment(pdb_path, ref_path, out_path):
    import pymol
    pymol.cmd.load(pdb_path, "mobile")
    pymol.cmd.load(ref_path, "ref")
    pymol.cmd.select("ref_chainA", "ref and chain A")
    pymol.cmd.select("mobile_chainA", "mobile and chain A")
    pymol.cmd.align("mobile_chainA", "ref_chainA")
    pymol.cmd.save(out_path, "mobile")




def load_openeye_mol(pdb_path):
    ifs = oechem.oemolistream()
    ifs.SetFormat(oechem.OEFormat_PDB)
    ifs.open(pdb_path)
    mols = []
    i = 0
    for mol in ifs.GetMolBases():
        i += 1
        mols.append(oechem.OEMol(mol))
    return mols[0]


# def align_pdb_to_reference(pdb_path, ref_path):

def superpose_proteins(reference_protein: oechem.OEMolBase,
                       fit_protein: oechem.OEMolBase,
                       residues: Iterable = tuple(),
                       chain_id: str = " ",
                       insertion_code: str = " ",
                       ) -> oechem.OEMolBase:
    from openeye import oespruce


    # do not modify input
    superposed_protein = fit_protein.CreateCopy()

    # site_residues = f"{residue[:3]}:{residue[3:]}"
    # set superposition method
    options = oespruce.OESuperpositionOptions()
    if len(residues) == 0 and chain_id == " " and insertion_code == " ":
        options.SetSuperpositionType(oespruce.OESuperpositionType_Global)
    elif chain_id != " ":
        options.SetSuperpositionType(oespruce.OESuperpositionType_Site)
        hv = oechem.OEHierView(superposed_protein)
        for chain in hv.GetChains():
            print(chain.GetChainID())
        res_strs = [f"{residue.GetResidueName()}:{residue.GetResidueNumber()}:{insertion_code}:{chain_id}" for residue
                    in hv.GetResidues() if residue.GetResidueName() == "CYS"]
        options.SetSiteResidues(res_strs)
        # for residue in res_strs:
        #     # print(f"{residue[:3]}:{residue[3:]}:{insertion_code}:{chain_id}")
        #     options.AddSiteResidue()

    else:
        options.SetSuperpositionType(oespruce.OESuperpositionType_Site)
        for residue in residues:
            options.AddSiteResidue(f"{residue[:3]}:{residue[3:]}:{insertion_code}:{chain_id}")

    # perform superposition
    # superposition = oespruce.OEStructuralSuperposition(
    #     reference_protein, superposed_protein, options
    # )
    superposition = oespruce.OESecondaryStructureSuperposition(
        reference_protein, superposed_protein, options
    )
    # for site_res in options.GetSiteResidues():
        # print(site_res)
    # print(superposition.GetRegion())
    print(superposition.GetFitChains())
    print(superposition.GetRefChains())
    superposition.Transform(superposed_protein)

    return superposed_protein


def align_all_pdbs(pdb_list, pdb_dir_path, ref_path=None, ref_name=None):
    if not ref_path:
        ref = pdb_list[0]
        ref_path = os.path.join(pdb_dir_path, f'rcsb_{ref}.pdb')
    else:
        ref = ref_name
    ref_mol = load_openeye_mol(ref_path)

    ofs = oechem.oemolostream()
    for pdb in pdb_list:
        pdb_path = os.path.join(pdb_dir_path, f'rcsb_{pdb}.pdb')
        pdb_mol = load_openeye_mol(pdb_path)
        new_pdb_path = os.path.join(pdb_dir_path, f"{pdb}_aligned_to_{ref}.pdb")
        ofs.open(new_pdb_path)
        print(f"Aligning {pdb_mol.GetTitle()} to {pdb_mol.GetTitle()} on chain A")
        aligned_mol = superpose_proteins(ref_mol,
                                         pdb_mol,
                                         chain_id="A"
                                         )

        print(f"Saving aligned molecule to {new_pdb_path}")
        oechem.OEWriteMolecule(ofs, aligned_mol)

def loading_openeye(molecule: oechem.OEMolBase):
    hv = oechem.OEHierView(molecule)
    chains = [chain for chain in hv.GetChains()]
    print(len(chains))

if __name__ == '__main__':
    pdb_list = load_pdbs_from_yaml('mers-structures.yaml')
    pdb_dir_path = '/Users/alexpayne/lilac-mount-point/mers-structures'
    # ref_path = '/Users/alexpayne/lilac-mount-point/fragalysis/extra_files/reference.pdb'
    ref_path = pdb_dir_path + "/rcsb_4RSP.pdb"
    # download_PDBs(pdb_list, pdb_path)
    # align_all_pdbs(pdb_list, pdb_dir_path,
    #                ref_path=ref_path,
    #                ref_name='4RSP')
    # mol = load_openeye_mol()
    # loading_openeye(mol)

    # ref_path = os.path.join(pdb_dir_path, f'rcsb_{ref}.pdb')
    # ref = "frag_ref"
    # for pdb in pdb_list:
    #     print(pdb)
    #     pdb_path = os.path.join(pdb_dir_path, f'rcsb_{pdb}.pdb')
    #     out_path = os.path.join(pdb_dir_path, f"{pdb}_aligned_to_{ref}.pdb")
    #     mdanalysis_alignment(pdb_path,
    #                          ref_path,
    #                          out_path)

    pdb_path = pdb_dir_path + "/rcsb_4YLU.pdb"
    new_pdb_path = os.path.join(pdb_dir_path, "4YLU_aligned_to_4RSP_pymol.pdb")
    pymol_alignment(pdb_path, ref_path, new_pdb_path)