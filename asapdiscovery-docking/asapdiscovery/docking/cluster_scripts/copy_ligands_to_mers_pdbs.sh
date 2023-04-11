#!/bin/bash
#BSUB -J copy_ligands_to_mers_pdbs
#BSUB -oo copy_ligands_to_mers_pdbs.out
#BSUB -eo copy_ligands_to_mers_pdbs.stderr
#BSUB -n 32
#BSUB -q cpuqueue
#BSUB -R rusage[mem=8]
#BSUB -W 168:00
source ~/.bashrc
conda activate ad-3.9
copy-ligands-to-new-structures \
-l "/lila/data/chodera/asap-datasets/mpro_fragalysis_2022_10_12/Mpro_combined_labeled.sdf" \
-p '/lila/data/chodera/asap-datasets/current/mers_01_prepped_pdbs_v2/*/*_prepped_receptor_0.pdb' \
-o "/lila/data/chodera/asap-datasets/mers_fauxalysis/20230411_prepped_for_fauxalysis" \
-n 32 \
--debug_num 32 \
--by_compound

echo Done
date