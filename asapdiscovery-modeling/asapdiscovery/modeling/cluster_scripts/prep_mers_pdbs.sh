#!/usr/bin/env bash
# Set walltime limit
#BSUB -W 1:00

# Set output file
#BSUB -o  log_files/prep_mers_pdbs.out

#BSUB -J prep_mers_pdbs

# Set error file
#BSUB -e log_files/prep_mers_pdbs.stderr

# Specify node group
#BSUB -q cpuqueue

# nodes: number of nodes
#BSUB -n 10 -R "rusage[mem=8]"


source ~/.bashrc
conda activate mers-docking

prep-mers-pdbs -n 10 \
-d /data/chodera/asap-datasets/mers_fauxalysis/mers_pdb_download \
-r /data/chodera/asap-datasets/mpro_fragalysis_2022_10_12/extra_files/reference.pdb \
-o /data/chodera/asap-datasets/mers_fauxalysis/20230308_prepped_mers_pdbs \
-l /data/chodera/asap-datasets/rcsb_spruce.loop_db \
-s "../../../../metadata/mpro_mers_seqres.yaml" \
--protein_only
