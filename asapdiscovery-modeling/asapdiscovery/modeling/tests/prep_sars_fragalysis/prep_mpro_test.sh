#!/bin/bash
# first, remove previous prepped directory, otherwise check_completed will cause the function to stop
rm -r prepped

# run protein prepped script
python ../../scripts/prep_proteins.py \
-d aligned \
-x metadata.csv \
-o prepped \
-n 1 \
-s ../../../../../metadata/mpro_sars2_seqres.yaml \
--include_non_Pseries # this is actually needed here because it appears that
                      # my current code returns the B dataset if remove_duplicates is true
