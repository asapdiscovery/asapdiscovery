# I know we'd rather use pytest, but this was much faster to write
run-docking-oe \
-l './inputs/2022_12_02_fragalysis_correct_bond_orders_220_P_structures.sdf' \
-r './inputs/Mpro-P0008_0A_ERI-UCB-ce40166b-17/*.oedu' \
-o . \
-n 2 \
--omega \
--relax clash \
--debug_num 2
