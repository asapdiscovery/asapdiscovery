#!/usr/bin/env bash
# Set walltime limit
#BSUB -W 0:15

# Set output file
#BSUB -o  log_files/download_mers_pdbs.out

#BSUB -J download_mers_pdbs

# Set error file
#BSUB -e log_files/download_mers_pdbs.stderr

# Specify node group
#BSUB -q cpuqueue

# nodes: number of nodes
#BSUB -n 1 -R "rusage[mem=8]"


source ~/.bashrc
conda activate mers-docking

python download_pdbs.py \
-d /data/chodera/asap-datasets/mers_fauxalysis/mers_pdb_download \
--pdb_yaml_path ../../../../metadata/mers-structures.yaml \
-t cif1
