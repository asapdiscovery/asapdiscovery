"""
Script to convert a single pdb file with multiple ligands all named LIG in
one crystal structure from Diamond to multiple pdb file each containing one
ligand named LIG bound to the original protein crystall structure and a sdf
file that contains all of the ligands. All the pdb files would be within a
directory within the directory of the original protein called output.
The sdf file containing all the ligands in original file will be also in that
directory named Combined_ligands.sdf.

Minimal example usage:
python cdd_to_schema.py \
-i multiple_ligand_bound.pdb \
-d directory/
"""

import argparse
import os
from pathlib import Path

import networkx as nx
from pymol import cmd

from asapdiscovery.data.openeye import (
    combine_protein_ligand,
    load_openeye_pdb,
    save_openeye_pdb,
    save_openeye_sdfs,
)


# Arguments define the input file and the output file
def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-d", required=True, help="Directory of input and output files."
    )
    parser.add_argument("-i", required=True, help="PDB file input.")

    return parser.parse_args()


# The function to isolate ligands in original file and separate out individual ligands
# Each ligand saved in individual pdb file
def split_ligands(directory, input_file):
    # Where original file is from
    local_path = Path(directory)
    # Where ending strucutures will be put into
    os.mkdir(directory + "output/")
    output_path = Path(directory + "output/")

    # Load structure into pymol
    cmd.load(local_path / input_file, "complex")

    # Function that saves residues named "LIG" to a PDB file
    def save_ligand_residues(structure_name, output_file):
        # Select residues named "LIG"
        selection = f"{structure_name} and resn LIG"

        # Save the selected residues to a PDB file
        cmd.save(output_file, selection, format="pdb")

    # Save all the ligands into one file
    save_ligand_residues("complex", local_path / "ligand_only.pdb")

    # Make a Graph out of pdb CONECT record
    ligands = local_path / "ligand_only.pdb"
    G = nx.Graph()
    with open(ligands) as f:
        for line in f:
            if line.startswith("CONECT"):
                # Extract atom numbers
                atom_numbers = [int(x.strip()) for x in line.split()[1:]]

                # Add edges to the graph
                for atom1 in atom_numbers[1:]:
                    G.add_edge(atom_numbers[0], atom1)
    # Delete previous pymol structure before proceeding so no confusion
    cmd.delete("complex")
    # Load only ligands into Pymol
    cmd.load(ligands, "ligands")
    # Remove this pdb since no longer in use
    os.remove(ligands)

    # Function to save each group connected atoms (a ligand) into individual files (with bonds)
    def save_subset_with_conect(structure_name, atom_ids, output_file):
        # Select atoms by IDs
        selection = f"{structure_name} and id {'+'.join(map(str, atom_ids))}"

        # Create a new object containing only the selected atoms
        cmd.create("selected_atoms", selection)

        # Save the selection with CONECT records to a PDB file
        cmd.save(output_file, "selected_atoms", format="pdb")

        # Delete the temporary object
        cmd.delete("selected_atoms")

    # Know which atom is from which ligand based on the graph constructed
    # List of the different ligands as {} of atoms
    atom_list = list(nx.connected_components(G))

    for i, atom_ids in enumerate(atom_list):
        # Save the selected subset with CONECT records to a PDB file
        save_subset_with_conect("ligands", atom_ids, local_path / f"lig_{i}.pdb")

    # Load pdb files and save all singled out ligands to singular combined sdf
    ligs = []
    for lig_pdb_file in local_path.glob("lig_*.pdb"):
        lig = load_openeye_pdb(lig_pdb_file)
        lig.SetTitle(lig_pdb_file.stem)
        ligs.append(lig)
    save_openeye_sdfs(ligs, output_path / "Combined_ligs.sdf")

    # Put each individual ligands back in protein in original orientation
    # Isolate protien in pymol and save to a pdb file
    cmd.load(local_path / input_file, "complex")
    cmd.select("protein", "complex and polymer.protein")
    cmd.save(local_path / "protein.pdb", "protein")
    cmd.delete("complex")

    # Load in the protein into openeye
    protein_file = local_path / "protein.pdb"
    protein = load_openeye_pdb(protein_file)

    # Load in each of the ligands and combine them with the protein
    # Save in folder called output within current path
    for lig_pdb_file in local_path.glob("lig_*.pdb"):
        # Load in the ligand pdb into openeye
        lig = load_openeye_pdb(lig_pdb_file)
        # Put it back
        lig_protein = combine_protein_ligand(protein, lig)
        # Save in file
        output_file = str(lig_pdb_file.stem) + "_protein.pdb"
        save_openeye_pdb(lig_protein, output_path / output_file)
        # Delete the ligand pdb since no longer useful
        os.remove(lig_pdb_file)

    # Remove the pdb file with just the protein structure
    os.remove(protein_file)


def main():
    args = get_args()

    # The function would be
    split_ligands(args.d, args.i)


if __name__ == "__main__":
    main()
