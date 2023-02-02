#!/usr/bin/env bash
#BSUB -J run_single_simulation
#BSUB -oo log_files/MProP0009_Sub.out
#BSUB -e log_files/MProP0009_Sub.stderr
#BSUB -n 1
#BSUB -R rusage[mem=4]
#BSUB -q gpuqueue
#BSUB -W 12:00

source ~/.bashrc
conda activate docking

python MproP0009.py \
-i /data/chodera/asap-datasets/full_frag_prepped_mpro_12_2022/Mpro-P0009_0A_MAT-POS-f2460aef-1/prepped_receptor_0.pdb \
-o .
echo date
echo done
