## load pdb file of interest
load /Users/alexpayne/Scientific_Projects/mers-drug-discovery/Mpro-paper-ligand/aligned/Mpro-P2295_0A/Mpro-P2295_0A_bound.pdb

## use selections in yaml files to select and color protein
select_from_file /Users/alexpayne/Scientific_Projects/covid-moonshot-ml/data/sars2.yaml, Mpro-P2295_0A_bound

color_from_file /Users/alexpayne/Scientific_Projects/covid-moonshot-ml/data/color_selection.yaml, Mpro-P2295_0A_bound

## set a bunch of stuff for visualization
set surface_color, grey90
bg_color white
hide everything
show cartoon
show surface
set cartoon_color, grey
set transparency, 0.3
show sticks, *_ligand
color pink, elem C and *_ligand
show sticks, *_P*
set_view (\
    -0.612729967,   -0.508448660,   -0.605001509,\
     0.194971263,   -0.839131176,    0.507754028,\
    -0.765855253,    0.193166196,    0.613299787,\
     0.000347150,    0.000152293, -105.557266235,\
     9.282041550,    0.385488749,   22.388103485,\
    96.971336365,  114.121208191,  -20.000000000 )
