#!/bin/bash
#BSUB -J calculate_RMSD
#BSUB -o log_files/calculate_RMSD_of_docked_ligands.out
#BSUB -e log_files/calculate_RMSD_of_docked_ligands.stderr
#BSUB -n 16
#BSUB -q cpuqueue
#BSUB -R rusage[mem=4]
#BSUB -W 1:00
source ~/.bashrc
conda activate ad-3.9
calculate-RMSD-of-docked-ligands \
-sdf /data/chodera/asap-datasets/retro_docking/sars_fragalysis_retrospective/20221208/combined.sdf \
-o /data/chodera/asap-datasets/retro_docking/sars_fragalysis_retrospective/20221208_analysis \
-r '/data/chodera/asap-datasets/full_frag_prepped_mpro_20230122/*/*.sdf' \
-n 16

echo Done
date
