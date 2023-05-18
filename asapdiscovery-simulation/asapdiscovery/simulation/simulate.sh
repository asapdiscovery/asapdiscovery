#!/bin/bash
#BSUB -P "asap"
#BSUB -J "vanilla"
#BSUB -n 1
#BSUB -R rusage[mem=16]
#BSUB -R span[hosts=1]
#BSUB -q gpuqueue
#BSUB -sp 25 # low priority. default is 12, max is 25
#BSUB -gpu num=1:j_exclusive=yes:mode=shared
#BSUB -W 10:00
#BSUB -o out_%J_%I.stdout
#BSUB -eo out_%J_%I.stderr
#BSUB -L /bin/bash

set -e # exit on error

source ~/.bashrc
OPENMM_CPU_THREADS=1

echo "changing directory to ${LS_SUBCWD}"
cd $LS_SUBCWD
conda activate asap-simulate

# Report node in use
hostname

# Report CUDA info
env | sort | grep 'CUDA'

# Report GPU info
nvidia-smi -L
nvidia-smi --query-gpu=name --format=csv

# simulate for 5 ns.
python simulate.py --receptor protein.pdb --ligand ligand.sdf --nsteps 2500000 --selection "not water" --minimized minimized.pdb --xtctraj trajectory.xtc --final final.pdb

# make the gif for easy viewing.
python traj_to_gif.py --system minimized.pdb --traj trajectory.xtc --gif trajectory.gif --smooth=5 --pse_share --contacts

# make an interactive protein-ligand interaction 2D diagram
conda activate prolif
python run_prolif.py minimized.pdb trajectory.xtc
