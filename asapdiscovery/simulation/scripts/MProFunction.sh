#!/usr/bin/env bash
#BSUB -J MPro_single_simulation
#BSUB -oo log_files/MProFunction.out
#BSUB -e log_files/MProFunction.stderr
#BSUB -n 2
#BSUB -gpu 'num=1'
#BSUB -R 'span[hosts=1]'
#BSUB -q gpuqueue
#BSUB -W 12:00

source ~/.bashrc
conda activate asap-simulation

python /data/chodera/lemonsk/covid-moonshot-ml/asapdiscovery/simulation/scripts/MProFunction.py -n 50000 \
-p /data/chodera/asap-datasets/current/sars_01_prepped_v3/Mpro-P0008_0A_ERI-UCB-ce40166b-17/Mpro-P0008_0A_ERI-UCB-ce40166b-17_protein.pdb \
-l /data/chodera/asap-datasets/current/sars_01_prepped_v3/Mpro-P0008_0A_ERI-UCB-ce40166b-17/Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf \
-o /data/chodera/lemonsk/asap-datasets/MPro_Simulations/
echo date
echo done
