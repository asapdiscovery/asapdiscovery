# I know we'd rather use pytest, but this was much faster to write
run-docking-oe \
-l './inputs/Mpro_combined_labeled.sdf' \
-r './inputs/Mpro-P0008_0A_ERI-UCB-ce40166b-17/*.oedu' \
-o './outputs/.' \
-n 2 \
--omega \
--relax clash \
--debug_num 4
