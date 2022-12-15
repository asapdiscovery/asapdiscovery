#!/usr/bin/env bash
# Set walltime limit
#BSUB -W 1:00

# Set output file
#BSUB -o  log_files/prep_mers_pdb_download.out

#BSUB -J prep_mers_pdb_download

# Set error file
#BSUB -e log_files/prep_mers_pdb_download.stderr

# Specify node group
#BSUB -q cpuqueue

# nodes: number of nodes
#BSUB -n 10 -R "rusage[mem=8]"


source ~/.bashrc
conda activate mers-docking
#python ../asapdiscovery/docking/scripts/prep_proteins.py -n 10 \
#-d /data/chodera/asap-datasets/mers_pdb_download \
#-p ../data/mers-structures-dimers.yaml \
#-r ~/fragalysis/extra_files/reference.pdb \
#-l ~/rcsb_spruce.loop_db \
#-o /data/chodera/asap-datasets/mers_prepped_structures \
#-s ../data/mpro_mers_seqres.yaml \
#--protein_only

python prep_proteins.py -n 10 \
-d /data/chodera/asap-datasets/mers_pdb_download \
-p ../../../metadata/mers-structures-dimers.yaml \
-r /data/chodera/asap-datasets/mpro_fragalysis_2022_10_12/extra_files/reference.pdb \
-l /data/chodera/asap-datasets/rcsb_spruce.loop_db \
-o /data/chodera/asap-datasets/mers_fauxalysis/mers_prepped_structures_dimers_only \
-s ../../../metadata/mpro_mers_seqres.yaml \
--protein_only