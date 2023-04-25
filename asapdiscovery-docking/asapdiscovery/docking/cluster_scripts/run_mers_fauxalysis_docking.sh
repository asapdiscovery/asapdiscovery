#!/bin/bash
#BSUB -J run_mers_fauxalysis_docking
#BSUB -oo run_mers_fauxalysis_docking.out
#BSUB -eo run_mers_fauxalysis_docking.stderr
#BSUB -n 72
#BSUB -q cpuqueue
#BSUB -R rusage[mem=2]
#BSUB -W 168:00
source ~/.bashrc
conda activate ad-3.9
run-docking-oe \
-l /lila/data/chodera/asap-datasets/mpro_fragalysis_2022_10_12/Mpro_combined_labeled.sdf \
-r '/lila/data/chodera/asap-datasets/mers_fauxalysis/20230411_prepped_for_fauxalysis/*/*_prepped_receptor_0.oedu' \
-o /lila/data/chodera/asap-datasets/retro_docking/sars_fragalysis_retrospective/20230330/ \
-n 72 \
--omega \
--relax clash
echo Done
date
