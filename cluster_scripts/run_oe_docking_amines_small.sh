
#!/bin/bash
#BSUB -J run_oe_docking_amines_small
#BSUB -R span[hosts=1]
#BSUB -oo log_files/run_oe_docking_amines_small.out
#BSUB -n 4
#BSUB -R rusage[mem=4]
#BSUB -W 24:00
conda activate mers-docking2
python ~/covid-moonshot-ml/asapdiscovery/docking/scripts/run_docking_oe.py \
-l ~/asap-datasets/amines_to_dock_small.sdf \
-r '~/asap-datasets/full_frag_prepped_dus_seqres/*/prepped_receptor.oedu' \
-s /lila/data/chodera/asap-datasets/amines_small_mcs/mcs_sort_index.pkl \
-o /lila/data/chodera/asap-datasets/alex_test \
-n 32 \
-t 10
echo done
