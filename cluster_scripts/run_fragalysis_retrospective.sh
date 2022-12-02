#!/bin/bash
#BSUB -J test_run_oe_docking
#BSUB -o log_files/run_fragalysis_retrospective.out
#BSUB -e log_files/run_fragalysis_retrospective.stderr
#BSUB -n 64
#BSUB -q cpuqueue
#BSUB -R rusage[mem=4]
#BSUB -W 72:00
source ~/.bashrc
conda activate mers-docking
python ~/covid-moonshot-ml/asapdiscovery/docking/scripts/run_docking_oe.py \
-l ~/asap-datasets/mpro_fragalysis_2022_10_12/2022_12_02_fragalysis_correct_bond_orders_220_P_structures.sdf \
-r '/lila/data/chodera/asap-datasets/prospective/prepped_mpro_structures_fragalysis/*/prepped_receptor.oedu' \
-o ~/asap-datasets/retro_docking/sars_fragalysis_retrospective/20221202/ \
-n 64
echo Done
date