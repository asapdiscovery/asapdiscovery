rm -rf ./outputs/test_generate_docking_input_csv
python ../scripts/generate_docking_input_csv.py \
-g "/Users/alexpayne/Scientific_Projects/covid-moonshot-ml/asapdiscovery-docking/asapdiscovery/docking/tests/inputs/prepped_mers_receptors/*.oedu" \
-o ./outputs/test_generate_docking_input_csv/test_generate_docking_input.csv \
--protein_regex "rcsb_([A-Za-z0-9]{4})" \
--ligand_name "my_ligand" \
--split_by_n_rows 2
#--split_by_protein \
#--ligand_regex "\/([\w-]*)_rcsb" \
