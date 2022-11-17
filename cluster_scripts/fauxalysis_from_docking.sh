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

# nodes: number of nodes
#BSUB -n 1 -R "rusage[mem=8]"


source ~/.bashrc
conda activate mers-docking
python ../asapdiscovery/docking/scripts/fauxalysis_from_docking.py \
-c /lila/data/chodera/asap-datasets/posit_hybrid_no_relax_keep_water_frag/best_results.csv \
-i /lila/data/chodera/kaminowb/stereochemistry_pred/mers/mers_fragalysis/posit_hybrid_no_relax_keep_water \
-f /lila/data/chodera/kaminowb/stereochemistry_pred/fragalysis/aligned \
-o /lila/data/chodera/asap-datasets/posit_hybrid_no_relax_keep_water_frag