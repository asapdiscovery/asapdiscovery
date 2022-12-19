
#!/bin/bash
#BSUB -J prep_full_mpro_AP
#BSUB -R span[hosts=1]
#BSUB -oo log_files/prep_full_mpro_AP.out
#BSUB -e log_files/prep_full_mpro_AP.stderr
#BSUB -n 1
#BSUB -R rusage[mem=4]
#BSUB -W 24:00
mkdir -p /data/chodera/asap-datasets/full_frag_prepped_mpro_20221219/
source ~/.bashrc
conda activate mers-docking
python /data/chodera/paynea/covid-moonshot-ml/asapdiscovery/docking/scripts/prep_proteins.py \
-d /data/chodera/asap-datasets/mpro_fragalysis_2022_10_12/aligned/ \
-x /data/chodera/asap-datasets/mpro_fragalysis_2022_10_12/extra_files/Mpro_compound_tracker_csv.csv \
-o /data/chodera/asap-datasets/full_frag_prepped_mpro_20221219/ \
-l /data/chocera/asap-datasets/rcsb_spruce.loop_db \
-n 1 \
-s /data/chodera/paynea/covid-moonshot-ml/metadata/mpro_sars2_seqres.yaml \
--include_non_Pseries
echo date
echo done
