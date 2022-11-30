
#!/bin/bash
#BSUB -J test_run_oe_docking
#BSUB -R span[hosts=1]
#BSUB -o log_files/test_run_oe_docking.out
#BSUB -e log_files/test_run_oe_docking.stderr
#BSUB -n 10
#BSUB -q cpuqueue
#BSUB -R rusage[mem=4]
#BSUB -W 00:20
source ~/.bashrc
conda activate mers-docking
python ~/covid-moonshot-ml/asapdiscovery/docking/scripts/run_docking_oe.py \
-l ~/asap-datasets/test_run_oe/ligand.sdf \
-r '~/asap-datasets/full_frag_prepped_dus_seqres/*/prepped_receptor.oedu' \
-s /lila/data/chodera/asap-datasets/amines_small_mcs/mcs_sort_index.pkl \
-o ~/asap-datasets/test_run_oe/ \
-n 10 \
-t 10
echo done
