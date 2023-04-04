
#!/bin/bash
#BSUB -J prep_full_mpro
#BSUB -R span[hosts=1]
#BSUB -oo log_files/prep_full_mpro.out
#BSUB -cwd /lila/data/chodera/kaminowb/stereochemistry_pred/mers
#BSUB -n 32
#BSUB -R rusage[mem=4]
#BSUB -W 24:00
mkdir -p /lila/data/chodera/kaminowb/stereochemistry_pred/mers//asap-datasets//full_frag_prepped_mpro_12_2022/
PYTHONPATH=${PYTHONPATH}:$(readlink -f /lila/data/chodera/kaminowb/stereochemistry_pred/mers//covid-moonshot-ml/) \
python /lila/data/chodera/kaminowb/stereochemistry_pred/mers//covid-moonshot-ml//asapdiscovery/docking/scripts//prep_proteins.py \
-d /lila/data/chodera/kaminowb/stereochemistry_pred/mers//asap-datasets//mpro_fragalysis_2022_10_12//aligned/ \
-x /lila/data/chodera/kaminowb/stereochemistry_pred/mers//asap-datasets//mpro_fragalysis_2022_10_12//extra_files/Mpro_compound_tracker_csv.csv \
-o /lila/data/chodera/kaminowb/stereochemistry_pred/mers//asap-datasets//full_frag_prepped_mpro_12_2022/ \
-l /lila/home/kaminowb/.openeye/rcsb_spruce.loop_db \
-n 32 \
-s /lila/data/chodera/kaminowb/stereochemistry_pred/mers//covid-moonshot-ml//metadata/mpro_sars2_seqres.yaml
echo done
