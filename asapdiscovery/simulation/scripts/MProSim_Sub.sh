#!/usr/bin/env bash
#BSUB -J MPro_single_simulation
#BSUB -oo log_files/MProSim_Sub.out
#BSUB -e log_files/MProSim_Sub.stderr
#BSUB -n 1
#BSUB -R rusage[mem=4]
#BSUB -q gpuqueue
#BSUB -W 12:00

source ~/.bashrc
conda activate /home/lemonsk/miniconda3/envs/asap-simulation

python /data/chodera/lemonsk/covid-moonshot-ml/asapdiscovery/simulation/scripts/MProSimulation.py -n 12 \
-p /data/chodera/lemonsk/asap-datasets/openmm_setup_processed/prepped_receptor_0-processed.pdb \
-l /data/chodera/lemonsk/covid-moonshot-ml/asapdiscovery/simulation/tests/inputs/MAT-POS-f2460aef-1.sdf \
-o /data/chodera/lemonsk/asap-datasets/MPro_Simulations
echo date
echo done
