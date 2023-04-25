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
dir='/lila/data/chodera/asap-datasets/mers_fauxalysis/20230411_prepped_for_fauxalysis/*/'
i=$LSB_JOBINDEX
array=( $(ls -d $dir) )
f=${array[$i]}
realpath $f
run-self-docking-oe \
-r $f'/*_prepped_receptor_0.oedu' \
-o  /lila/data/chodera/asap-datasets/mers_fauxalysis/20230425_docked_for_fauxalysis \
-n 1 \
--omega \
--relax clash \
--debug_num 1
echo Done
date
