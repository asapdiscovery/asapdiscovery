# Generating Structures with ColabFold

This directory contains scripts for interfacing with ColabFold to generate
structures of proteins that we don't yet have crystal structures for, to be
used as inputs for the other `asapdiscovery` pipelines.

## Environment Prep
Create your environment for these scripts using the instructions from the
[ColabFold repo](https://github.com/sokrypton/ColabFold/):
```bash
pip install "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold"
pip install -q "jax[cuda]>=0.3.8,<0.4" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# For template-based predictions also install kalign and hhsuite
conda install -c conda-forge -c bioconda kalign2=2.04 hhsuite=3.3.0
```

## General Pipeline
There are three steps for now:
1. Prepare Fragalysis structures for use as ColabFold templates
2. Prepare sequence input for structures to generate
3. Generate structures from ColabFold

### Fragalysis Prep
To prepare Fragalysis for use as ColabFold templates, all non-protein atoms
need to be stripped out. Using the `copy_and_clean_frag.sh` script:
```bash
./scripts/copy_and_clean_frag.sh /path/to/fragalysis/aligned /path/to/output/
```

### Sequence Prep
The ColabFold script takes a CSV file as input with two columns: `id`, and
`sequence`. For the script included in this repo, the names in the `id` column
should have a blank `{}` in them, which will be replaced by the random seed
that is used to generate the protein. This is to prevent ColabFold from thinking
it has already generated the structure when the random seed is different. We
have prepared the file `./metadata/MERS_Mpro_input.csv`, but this file can also
be generated from the results of a `BLASTP` search using the `blast_search.py`
script:
```bash
./scripts/blast_search.py -if query_seq.fasta -o blast_colabfold_in.csv
```

### Structure Generation
The `run_colabfold.sh` script in this repo will generate a random seed, fill
that random seed into the `id` column in the input CSV file, and then run
ColabFold. The next CLI arg to this script, which is optional, is the number of
structures to generate. Each structure will have a different random seed, which
allows different structures to be generated from the same input sequence. If
this arg is not provided, only one structure will be generated. There is also an
optional `--no-rand` flag, which fully disables the random seed usage, and
instead will use the default seed of `0`. Finally, any additional args that are
passed when calling this script will be passed directly to `colabfold_batch`.
Example usage to generate 10 different structures, each with a different random
seed:
```bash
# Using a fixed MERS Mpro sequence
./scripts/run_colabfold.sh ./metadata/MERS_Mpro_input.csv \
/path/to/templates/ /path/to/output/ 10
# Using a generated file from a BLASTP search
./scripts/run_colabfold.sh blast_colabfold_in.csv \
/path/to/templates/ /path/to/output/ 10
```
