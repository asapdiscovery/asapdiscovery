#!/bin/bash

################################################################################
## Script to prepare a fragalysis download for use as templates when running
##  colabfold
## Usage is ./copy_and_clean_frag.sh ./fragalysis/aligned/ ./templates/
################################################################################

in_dir=$1
out_dir=$2

echo $in_dir
echo $out_dir

## Loop through Mpro-P* structures
for fn in ${in_dir}/Mpro-P*/*bound.pdb; do
    ## Extract 4 numbers after P to use as the "PDB id" to make colabfold happy
    b=$(grep -Eo Mpro-P[0-9]{4} <<< $fn | head -n 1)
    b=${b#Mpro-P}
    out_fn=${out_dir}/${b}.pdb
    ## Only want to take the first copy we find of each structure
    [[ -f $out_fn ]] && continue
    echo $fn $b
    ## Remove all the HETATM lines
    grep -Fv HETATM $fn > $out_fn
done
