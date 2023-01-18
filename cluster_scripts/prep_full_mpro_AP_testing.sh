#!/bin/bash
#BSUB -J prep_full_mpro_AP_test
#BSUB -oo log_files/prep_full_mpro_AP_test.out
#BSUB -e log_files/prep_full_mpro_AP_test.stderr
#BSUB -n 32
#BSUB -R rusage[mem=4]
#BSUB -W 72:00
mkdir -p /data/chodera/asap-datasets/full_frag_prepped_mpro_20230118/
source ~/.bashrc
conda activate mers-docking
python /data/chodera/paynea/covid-moonshot-ml/asapdiscovery/docking/scripts/prep_proteins.py \
-d /data/chodera/asap-datasets/mpro_fragalysis_2022_10_12/aligned/ \
-x /data/chodera/asap-datasets/mpro_fragalysis_2022_10_12/extra_files/Mpro_compound_tracker_csv.csv \
-o /data/chodera/asap-datasets/full_frag_prepped_mpro_20230118/ \
-l /data/chodera/asap-datasets/rcsb_spruce.loop_db \
-n 32 \
-s /data/chodera/paynea/covid-moonshot-ml/metadata/mpro_sars2_seqres.yaml \
--include_non_Pseries
echo date
echo done
