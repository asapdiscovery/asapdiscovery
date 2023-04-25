#!/bin/zsh
rm -rf outputs/test_self_docking
python /Users/alexpayne/Scientific_Projects/covid-moonshot-ml/asapdiscovery-docking/asapdiscovery/docking/scripts/run_self_docking_oe.py \
-r "./inputs/multireceptor_docking_test/*.oedu" \
-o outputs/test_self_docking \
-n 1 \
--debug_num 1