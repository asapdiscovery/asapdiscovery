#!/usr/bin/env bash
#BSUB -J run_single_simulation
#BSUB -oo log_files/MProP0009_Sub.out
#BSUB -e log_files/MProP0009_Sub.stderr
#BSUB -n 1
#BSUB -R rusage[mem=4]
#BSUB -q gpuqueue
#BSUB -W 12:00

source ~/.bashrc
conda activate /home/lemonsk/miniconda3/envs/asap-simulation

python /data/chodera/lemonsk/covid-moonshot-ml/asapdiscovery/simulation/scripts/MproP0009.py \
-i /data/chodera/lemonsk/asap-datasets/openmm_setup_processed/prepped_receptor_0-processed.pdb \
-o /data/chodera/lemonsk/asap-datasets/prepped_mpro_P0009/ \
-l /data/chodera/lemonsk/covid-moonshot-ml/asapdiscovery/simulation/tests/inputs/MAT-POS-f2460aef-1.sdf \
-p /data/chodera/lemonsk/asap-datasets/openmm_setup_processed/prepped_receptor_0-processed.pdb
echo date
echo done
