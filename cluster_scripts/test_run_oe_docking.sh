
#!/bin/bash
#BSUB -J run_oe_docking_amines_small
#BSUB -R span[hosts=1]
#BSUB -oo log_files/test_run_oe_docking.out
#BSUB -n 1
#BSUB -R rusage[mem=4]
#BSUB -W 00:20
conda activate mers-docking2
python ~/covid-moonshot-ml/asapdiscovery/docking/scripts/run_docking_oe.py \
-l ~/asap-datasets/test_run_oe/ligand.sdf \
-r '~/asap-datasets/full_frag_prepped_dus_seqres/*/prepped_receptor.oedu' \
-s /lila/data/chodera/asap-datasets/amines_small_mcs/mcs_sort_index.pkl \
-o /lila/data/chodera/asap-datasets/alex_test \
-n 1 \
-t 10
echo done
