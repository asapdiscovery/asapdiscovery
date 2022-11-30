
#!/bin/bash
#BSUB -J run_oe_docking_amines_small
#BSUB -R span[hosts=1]
#BSUB -oo log_files/run_oe_docking_amines_small.out
#BSUB -cwd /lila/data/chodera/kaminowb/stereochemistry_pred/mers
#BSUB -n 32
#BSUB -R rusage[mem=4]
#BSUB -W 24:00
mkdir -p /lila/data/chodera/kaminowb/stereochemistry_pred/mers//asap-datasets//amines_small/
python /lila/data/chodera/kaminowb/stereochemistry_pred/mers//covid-moonshot-ml/asapdiscovery/docking/scripts//run_docking_oe.py \
-l /lila/data/chodera/kaminowb/stereochemistry_pred/mers//asap-datasets//amines_to_dock_small.sdf \
-r '/lila/data/chodera/kaminowb/stereochemistry_pred/mers//asap-datasets//full_frag_prepped_dus_seqres/*/prepped_receptor.oedu' \
-s /lila/data/chodera/kaminowb/stereochemistry_pred/mers//asap-datasets//amines_small_mcs/mcs_sort_index.pkl \
-o /lila/data/chodera/kaminowb/stereochemistry_pred/mers//asap-datasets//amines_small/ \
-n 32 \
-t 10
echo done
