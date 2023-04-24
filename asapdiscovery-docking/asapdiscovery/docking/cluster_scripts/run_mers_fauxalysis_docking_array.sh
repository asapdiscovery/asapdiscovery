#!/bin/bash
## Example Usage:
## bsub -J "run_mers_fauxalysis_docking[1-554]" < run_fragalysis_retrospective_array.sh

#BSUB -oo run_mers_fauxalysis_docking_%I.out
#BSUB -eo run_mers_fauxalysis_docking_%I.stderr
#BSUB -n 1
#BSUB -q cpuqueue
#BSUB -R rusage[mem=2]
#BSUB -W 2:00
#source ~/.bashrc
#conda activate ad-3.9
dir='/lila/data/chodera/asap-datasets/mers_fauxalysis/20230411_prepped_for_fauxalysis/*/'
i=%I
array=( $(ls -d $dir) )
f=${array[$i]}
realpath $f
#run-docking-oe \
#-l /lila/data/chodera/asap-datasets/mpro_fragalysis_2022_10_12/Mpro_combined_labeled.sdf \
#-r '/lila/data/chodera/asap-datasets/mers_fauxalysis/20230411_prepped_for_fauxalysis/*/*_prepped_receptor_0.oedu' \
#-o /lila/data/chodera/asap-datasets/retro_docking/sars_fragalysis_retrospective/20230330/ \
#-n 72 \
#--omega \
#--relax clash
#echo Done
#date
