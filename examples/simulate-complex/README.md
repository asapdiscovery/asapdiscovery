# Simulating a protein:ligand complex to generate a trajectory and movie

This script will run 

Inputs:  
* Protein PDB : must be fully protonated (e.g. Spruced Fragalysis structure)
* Ligand SDF (or other format RDKit can read) : does not need to be protonated

## Manifest
* `environment.yml` : install this conda environment to run this example
* `simulate.py` : script to generate simulation trajectory
* `traj_to_gif.py` : script to generate animated GIF of trajectory via pymol

## Installation

Install the conda environment:
```bash
mamba create -n asap-simulate -f environment.yml
```

## Example usage

Run a 1 ns simulation of a receptor:ligand complex:
```bash
python simulate.py --receptor "structures/Mpro-P1788_0A_bound-His41(+)-Cys145(-)-His163(+)-protein.pdb" --ligand "structures/Mpro-P1788_0A_bound-His41(+)-Cys145(-)-His163(+)-ligand.sdf" --nsteps 250000 --selection "not water" --minimized minimized.pdb --xtctraj trajectory.xtc --final final.pdb
```
Generate an animated GIF from the trajectory:
```bash
python traj_to_gif.py --system minimized.pdb --traj trajectory.xtc --gif trajectory.gif --smooth=5
```
To also generate contacts:
```bash
# Download David Koes show_contacts tool
wget https://raw.githubusercontent.com/Pymol-Scripts/Pymol-script-repo/master/plugins/show_contacts.py
# Generate trajectory with contacts
python traj_to_gif.py --system minimized.pdb --traj trajectory.xtc --gif trajectory.gif --smooth=5 --contacts
```