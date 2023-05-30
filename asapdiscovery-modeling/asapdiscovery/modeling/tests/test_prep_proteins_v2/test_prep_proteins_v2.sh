#!/bin/bash
# first, remove previous prepped directory, otherwise check_completed will cause the function to stop
rm -r prepped

# run protein prepped script
prep-proteins-v2 \
-csv ../test_prepare_protein_csv/to_prep.csv \
-r "../prep_mers_rcsb/inputs/reference.pdb" \
-o prepped

# try the version from glob
prep-proteins-v2 \
-csv ../test_prepare_protein_csv/to_prep_glob.csv \
-r "../prep_mers_rcsb/inputs/reference.pdb" \
--protein_only \
-o prepped_glob
