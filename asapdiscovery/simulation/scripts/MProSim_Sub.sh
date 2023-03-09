#!/usr/bin/env bash
#BSUB -J MPro_single_simulation
#BSUB -oo log_files/MProSim_Sub.out
#BSUB -e log_files/MProSim_Sub.stderr
#BSUB -n 1
#BSUB -n 2 -W 2:00 -q gpuqueue -gpu 'num=1' -R 'span[hosts=1]'
#BSUB -R rusage[mem=4]
#BSUB -q gpuqueue
#BSUB -W 12:00

source ~/.bashrc
conda activate asap-simulation

sed '/LIG/d' /data/chodera/asap-datasets/current/sars_01_prepped_v3/Mpro-P0008_0A_ERI-UCB-ce40166b-17/Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.pdb > protein_P0008.pdb

python /data/chodera/lemonsk/covid-moonshot-ml/asapdiscovery/simulation/scripts/MProSimulation.py -n 50 \
-p protein_P0008.pdb \
-l /data/chodera/asap-datasets/current/sars_01_prepped_v3/Mpro-P0008_0A_ERI-UCB-ce40166b-17/Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf \
-o /data/chodera/lemonsk/asap-datasets/MPro_Simulations/
echo date
echo done
