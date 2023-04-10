#!/bin/bash
# I know we'd rather use pytest, but this was much faster to write
rm -rf ./outputs/retro_docking_test
run-docking-oe \
-l './inputs/Mpro_combined_labeled.sdf' \
-r './inputs/Mpro-P0008_0A_ERI-UCB-ce40166b-17/*.oedu' \
-o './outputs/retro_docking_test' \
-n 2 \
--omega \
--relax clash \
--debug_num 4
