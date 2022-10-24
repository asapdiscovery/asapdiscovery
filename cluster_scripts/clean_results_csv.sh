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
python ../scripts/clean_results_csv.py \
-i /lila/data/chodera/kaminowb/stereochemistry_pred/mers/mers_fragalysis/posit_hybrid_no_relax_keep_water/all_results.csv \
-o /lila/data/chodera/paynea/posit_hybrid_no_relax_keep_water_frag \
-d