#!/bin/bash
# first, remove previous prepped directory, otherwise check_completed will cause the function to stop
rm -r to_prep.csv

# run protein prepped script
# the full path is needed in the directory in order to find the input files using the output prep csv
prepare-protein-csv \
-d /Users/alexpayne/Scientific_Projects/covid-moonshot-ml/asapdiscovery-modeling/asapdiscovery/modeling/tests/test_prepare_protein_csv/aligned \
-csv metadata.csv \
-o to_prep.csv \
--include_non_Pseries # this is actually needed here because it appears that
                      # my current code returns the B dataset if remove_duplicates is true

# try with protein glob instead
prepare-protein-csv \
--pdb_glob "/Users/alexpayne/Scientific_Projects/covid-moonshot-ml/asapdiscovery-modeling/asapdiscovery/modeling/tests/test_prepare_protein_csv/aligned/Mpro-P2660_0A/Mpro-P2660_0A_bound.pdb" \
--protein_only \
-o to_prep_glob.csv