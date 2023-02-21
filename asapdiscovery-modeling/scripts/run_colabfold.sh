#!/bin/bash

################################################################################
## Script to run colabfold_batch using a random seed
## Input CSV file should have a {} in the id column that will be filled with
##  the random seed to make sure different structure are generated
## Usage is ./run_colabfold.sh ./query.csv ./template_dir/ ./out_dir/
################################################################################

csv_fn=$1
template_dir=$2
out_dir=$3
n_gen=$4

echo csv $csv_fn
echo templates $template_dir
echo outputs $out_dir
echo samples $n_gen

[[ $n_gen == "" ]] && n_gen=1

for i in $(seq 1 $n_gen); do
    ## Generate random seed for structure generation
    rng_seed=$RANDOM
    echo running with seed $rng_seed

    ## Make temp CSV file with seed in name
    sed "s/{}/${rng_seed}/g" $csv_fn > tmp_in.csv

    ## Run colabfold with 3 recycles and 1 model
    colabfold_batch --random-seed $rng_seed --templates \
    --custom-template-path $template_dir --num-recycle 3 --num-models 1 \
    tmp_in.csv $out_dir && rm tmp_in.csv
done
