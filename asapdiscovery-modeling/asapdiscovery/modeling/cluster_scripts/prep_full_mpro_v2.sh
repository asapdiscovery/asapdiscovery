#!/bin/bash
#BSUB -J prep_full_mpro_v2
#BSUB -oo log_files/prep_full_mpro_v2.out
#BSUB -eo log_files/prep_full_mpro_v2.stderr
#BSUB -n 32
#BSUB -R rusage[mem=4]
#BSUB -W 12:00
source ~/.bashrc
conda activate ad-3.9
fragalysis-to-schema \
--metadata_csv /data/chodera/asap-datasets/mpro_fragalysis_2022_10_12/metadata.csv \
--aligned_dir /data/chodera/asap-datasets/mpro_fragalysis_2022_10_12/aligned/ \
-o /data/chodera/asap-datasets/full_frag_prepped_mpro_20230603/metadata \
--name_filter "Mpro-P" \
--drop_duplicates \

create-prep-inputs \
-i /data/chodera/asap-datasets/full_frag_prepped_mpro_20230603/metadata/fragalysis.csv \
-o /data/chodera/asap-datasets/full_frag_prepped_mpro_20230603/metadata \
