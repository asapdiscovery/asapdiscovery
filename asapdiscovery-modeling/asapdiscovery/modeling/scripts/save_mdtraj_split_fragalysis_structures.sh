#!/usr/bin/env bash
# Set walltime limit
#BSUB -W 1:00

# Set output file
#BSUB -o  %J.out

#BSUB -J "save-mdtraj-split-fragalysis-structures"

# Set error file
#BSUB -e %J.stderr

# Specify node group
#BSUB -q cpuqueue

# nodes: number of nodes
#BSUB -n 32 -R "rusage[mem=8]"


source ~/.bashrc
conda activate mers-docking
python ../scripts/save-combined-fragalysis-structures.py -n 32 \
-d /data/chodera/asap-datasets/current/sars_01_prepped_v3 \
-o /data/chodera/paynea/asap-datasets-ap/sars_01_prepped_v3_split
