#!/bin/bash
#BSUB -J run_oe_docking
#BSUB -o run_fragalysis_retrospective.out
#BSUB -e run_fragalysis_retrospective.stderr
#BSUB -n 2
#BSUB -q cpuqueue
#BSUB -R rusage[mem=4]
#BSUB -W 00:10
source ~/.bashrc
conda activate mers-docking
run-docking-oe \
-l /lila/data/chodera/asap-datasets/mpro_fragalysis_2022_10_12/2022_12_02_fragalysis_correct_bond_orders_220_P_structures.sdf \
-r '/lila/data/chodera/asap-datasets/full_frag_prepped_mpro_12_2022/*/prepped_receptor_0.oedu' \
-o /lila/data/chodera/asap-datasets/retro_docking/sars_fragalysis_retrospective/20230328/ \
-n 2 \
--omega \
--relax clash \
--debug_num 2
echo Done
date
