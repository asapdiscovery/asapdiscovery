#!/bin/bash
#BSUB -J run_oe_docking
#BSUB -oo run_fragalysis_retrospective.out
#BSUB -eo run_fragalysis_retrospective.stderr
#BSUB -n 32
#BSUB -q cpuqueue
#BSUB -R rusage[mem=16]
#BSUB -W 01:00
source ~/.bashrc
conda activate ad-3.9
run-docking-oe \
-l /lila/data/chodera/asap-datasets/mpro_fragalysis_2022_10_12/2022_12_02_fragalysis_correct_bond_orders_220_P_structures.sdf \
-r '/lila/data/chodera/asap-datasets/current/sars_01_prepped_v3/*/*_prepped_receptor_0.oedu' \
-o /lila/data/chodera/asap-datasets/retro_docking/sars_fragalysis_retrospective/20230328/ \
-n 32 \
--omega \
--relax clash \
--debug_num 32
echo Done
date
