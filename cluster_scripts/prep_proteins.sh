#!/usr/bin/env bash
# Set walltime limit
#BSUB -W 1:00

# Set output file
#BSUB -o  %J.out

#BSUB -J "prep_proteins"

# Set error file
#BSUB -e %J.stderr

# Specify node group
#BSUB -q cpuqueue

# nodes: number of nodes and GPU request
#BSUB -n 10 -R "rusage[mem=8]"
#BSUB -gpu "num=1:j_exclusive=yes:mode=shared"


source ~/.bashrc
conda activate mers-docking
python prep_proteins.py -d /data/chodera/asap-datasets/mers_pdb_download -p ../data/mers-structures.yaml -r ~/fragalysis/extra_files/reference.pdb -l ~/rcsb_spruce.loop_db -o /data/chodera/asap-datasets/mers_prepped_structures -s ../data/mpro_mers_seqres.yaml --protein_only
