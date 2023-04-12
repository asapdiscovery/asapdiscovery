#!/bin/bash
## Example usage:
## bsub -J "run_fragalysis_retrospective[1-58]" < run_fragalysis_retrospective_array.sh

#BSUB -J run_oe_docking
#BSUB -oo run_fragalysis_retrospective_%I.out
#BSUB -eo run_fragalysis_retrospective_%I.stderr
#BSUB -n 32
#BSUB -q cpuqueue
#BSUB -R rusage[mem=8]
#BSUB -W 1:00
source ~/.bashrc
conda activate ad-3.9
run-docking-oe \
-l "/lila/data/chodera/asap-datasets/current/sars_01_prepped_v3/sdf_lsf_array/"$LSB_JOBINDEX".sdf" \
-r '/lila/data/chodera/asap-datasets/current/sars_01_prepped_v3/*/*_prepped_receptor_0.oedu' \
-o '/lila/data/chodera/asap-datasets/retro_docking/sars_fragalysis_retrospective/20230411' \
-n 32 \
--omega \
--debug_num 1 \
--relax clash
echo Done
date
