#!/bin/bash
## Example Usage:
## bsub -J "run_mers_fauxalysis_docking[1-554]" < run_mers_fauxalysis_docking_array.sh

#BSUB -oo run_mers_fauxalysis_docking_%I.out
#BSUB -eo run_mers_fauxalysis_docking_%I.stderr
#BSUB -n 1
#BSUB -q cpuqueue
#BSUB -R rusage[mem=2]
#BSUB -W 2:00
source ~/.bashrc
conda activate ad-3.9
i=$LSB_JOBINDEX
array=( $(ls /lila/data/chodera/asap-datasets/mers_fauxalysis/20230411_prepped_for_fauxalysis/docking_input_csvs/*_docking_input.csv) )
f=${array[$i]}
echo "using $f as input"
run-self-docking-oe \
-csv "$f" \
-o  /lila/data/chodera/asap-datasets/mers_fauxalysis/20230425_docked_for_fauxalysis \
-n 1 \
--omega \
--relax clash \
--log_name "run_docking_oe.$LSB_JOBINDEX" \
--debug_num 1
echo Done
date
