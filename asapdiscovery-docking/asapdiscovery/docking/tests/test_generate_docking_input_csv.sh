python ../scripts/generate_self_docking_input_csv.py \
-g "/Users/alexpayne/Scientific_Projects/covid-moonshot-ml/asapdiscovery-docking/asapdiscovery/docking/tests/inputs/multireceptor_docking_test/*.oedu" \
-o ./outputs/test_generate_self_docking_input_csv/test_generate_self_docking_input.csv \
--protein_regex "rcsb_([A-Za-z0-9]{4})" \
--ligand_regex "\/([\w-]*)_rcsb" \
