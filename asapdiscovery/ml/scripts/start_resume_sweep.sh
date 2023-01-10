#!/bin/bash

config_file=$1
out_dir=$2
overwrite=$3

>&2 echo $config_file
>&2 echo $out_dir
>&2 echo $overwrite

sweep_id_fn=${out_dir}/sweep_id

## If overwrite isn't set and the file exists, just print it and exit
if [[ $overwrite == "" ]]; then
    if [[ -f $sweep_id_fn ]]; then
        sweep_id=$(cat $sweep_id_fn)
        if [[ $sweep_id == "" ]]; then
            >&2 echo "empty sweep_id file"
        else
            echo -n $sweep_id
            exit 0
        fi
    fi
fi

## Otherwise start sweep with the config file, store and print sweep id
wandb sweep $config_file 2>&1 | \
grep -F 'Created sweep with ID' | \
awk '{print $NF}'| \
tee $sweep_id_fn
