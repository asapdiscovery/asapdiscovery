# Intro
This document will show an example of running the ML pipeline all the way through,
from data prep to training to inference. This document is focused on running scripts
rather than interfacing with the API itself.

# Preparing Experimental Data
The first step in the pipeline is to download and parse the experimental data from CDD.
This will act as our labels when training, and will also give us the SMILES strings to
dock when generating our structural inputs.
## Download and Parse Data
First we will download and parse the CDD data. Both of these steps are handled in the
same script:
`asapdiscovery-data/asapdiscovery/data/scripts/download_moonshot_data.py`. Running this
script will require a CDD API token, which can either be stored in the `CDDTOKEN`
environment variable, or in a file that is then passed to the script. By default, the
script will download all Moonshot compounds that have fluorescence values for SARS-Cov-2
Mpro, but other searches can be specified using the CLI args.

As an example, we will download the default search, keeping all molecules that are
achiral or enantiopure and including semiquantitative data. We assume in this case that
the `CDDTOKEN` env variable has been set appropriately. This call also saves the
unfiltered and unprocessed file as a cache file to avoid repeated CDD download calls:
```bash
python /path/to/data/scripts/download_moonshot_data.py \
    -o cdd_achiral_enantiopure_semiquant_filt.csv \
    -cache cdd_achiral_enantiopure_semiquant_unfiltered.csv \
    --retain_achiral --retain_semiquant --retain_enantiopure
```
This will create two new files in the current directory:
`cdd_achiral_enantiopure_semiquant_filt.csv`, the filtered and processed download, and
`cdd_achiral_enantiopure_semiquant_unfiltered.csv`, the raw CDD download. It may also
be prudent to add the date downloaded somewhere in the filename, as this information is
not included anywhere else in the download.

## Convert Downloaded Data
Next we will convert the downloaded CSV file into a JSON file containing
`ExperimentalCompoundData` objects that will serve as part of the inputs to the ML
model. The script that handles this is
`asapdiscovery-data/asapdiscovery/data/scripts/cdd_to_schema.py`. This script doesn't
have too many options, and for most cases the following simple call will suffice:
```bash
python /path/to/data/scripts/cdd_to_schema.py \
    -i cdd_achiral_enantiopure_semiquant_filt.csv \
    -json cdd_achiral_enantiopure_semiquant_schema.json
```
Note that the input (`-i`) to this script should be the output (`-o`) from the previous
script. This will create a new file in the current directory
`cdd_achiral_enantiopure_semiquant_schema.json`, which contains the schema objects.

# Preparing Structural Data
We'll next need to prepare the structural data, which includes downloading the
Fragalysis protein structures, selecting which structures to dock each ligand to, and
finally running docking.
## Download Fragalysis data
**Note that this might not be working as Fragalysis has been having some issues.**

By default, this script will download the Fragalysis archive for the given target as a
.zip file (given by the `-o` CLI arg). To have the script also automatically extract the
arhive, pass the `-x` CLI option. Note that this will extract all files to the same
directory as the file given by `-o`. Make sure this is the directory that you want to be
your Fragalysis directory, otherwise you'll have to manually move files (and risk having
some of your existing files overwritten). In this example, we'll download the SARS-CoV-2
Mpro archive, first creating a new directory to hold everything.
```bash
mkdir -p mpro_fragalysis
python /path/to/data/scripts/download_fragalysis_data.py \
    -t mpro \
    -o ./mpro_fragalysis/mpro.zip \
    -x
```
The download .zip file can safely be deleted after this if desired:
```bash
rm ./mpro_fragalysis/mpro.zip
```

## Run Maximum Common Substructure Search
To figure out which Fragalysis structures to dock each ligand to, we run an MCS search
for all ligands against all the crystal structure ligands from Fragalysis. This will
rank all the structures based on the similarity between the crystal ligand and each
ligand to be docked. During the docking step we will determine how many of the top
structures each ligand will actually be docked to.

This example script call will use the downloaded CDD data and the downloaded and
extracted Fragalysis data from the previous steps. We will also run this step with 32
parallel processes. This number can be adjusted depending on your machine, but it is
recommended to parallelize this script as it can take a while otherwise.
```bash
mkdir -p mcs_results
python /path/to/docking/scripts/find_all_mcs.py \
    -exp cdd_achiral_enantiopure_semiquant_schema.json \
    -x ./mpro_fragalysis/extra_files/Mpro_compound_tracker_csv.csv \
    -x_dir ./mpro_fragalysis/aligned/ \
    -o ./mcs_results/ \
    -n 32 \
    -n_draw 0
```
The `-n_draw` option tells the script how many of the top matches for each compound to
draw the structure of, with the matched substructure highlighted. This can be useful if
you want to inspect how the MCS algorithm is working, but for production is not
necessary, so we use `-n_draw 0`.

## Prep OpenEye Design Units

## Run Docking
The final step in preparing the structural data is to run docking. There are a number of
options that can be configured, but this a minimal example that will dock each compound
to its top match from the MCS search. This step is again parallelized (`-n 32`), which
can be adjusted based on your machine.
```bash
mkdir -p docking_results
python /path/to/docking/scripts/run_docking_oe.py \
    -e cdd_achiral_enantiopure_semiquant_schema.json \
    -r ./prepped_receptors/*/prepped_receptor.oedu \
    -s ./mcs_results/mcs_sort_index.pkl \
    -o ./docking_results/ \
    -n 32
```

## Generate Docked PDB Files
The docking script generates a `docked.sdf` file for each compound:crystal structure
combination, which will contain all poses generated for that compound. Our final step
here will be to generate a PDB file for each of these ligand conformations that contain
the ligand placed in the crystal structure it was docked to. This is also parallelized
(`-w 12`) to speed things up. The output files for each `docked.sdf` file will be
generated in the same directory as the input file.

```bash
python /path/to/docking/scripts/make_docked_complexes_schema_v2.py \
    -d ./docking_results/ \
    -x ./mpro_fragalysis/ \
    -w 12
```

# Machine Learning
Finally we can get into the ML.

## Prepare ML Dataset
Before training the model, we will build the input `DockedDataset` object and cache it.
This step isn't strictly necessary, but it will save time during training and makes it
so that mulitple different training runs can be dispatched concurrently. Note that the
single quotes in the `-i` option are required to prevent the shell from expanding the
wildcard. The output generated by this call will work for any structure-based model. A
seperate dataset will need to be generated for graph-based models, but this same script
can be used (with a different arg passed for `-model`).
```bash
python /path/to/ml/scripts/build_dataset.py \
    -i './docking_results/*/*_bound.pdb' \
    -exp cdd_achiral_enantiopure_semiquant_schema.json \
    -o ./achiral_enantiopure_semiquant_struct_ds.pkl \
    -model schnet
```

## Train the Model
There are many different options for how this script can be called. This example shows
how to train a SchNet model (`-model schnet`) using all the data previously generated in
this example. We will train for 300 epochs (`-n_epochs 300`), and use temporal splitting
for our data (`--temporal`). Additionally, this example uses Weights & Biases to track
model training (`--wandb`). You should set up and initialize your W&B account according
to their docs before running this. We will log this to a `test` project (`--proj test`),
but you can obviously call this whatever you like. We will add a `PIC50Readout` block to
the model (`-pred_r pic50`) to allow training against experimental pIC50 values. Here we
use the experimental condition values from the Moonshot experiments
(`-sub 0.375 -km 9.5`), but these should be adjusted if you're using different data.
Finally, we use a step MSE loss function (`-loss step`), which prevents the model from
being penalized if the data it's trying to predict is outside the assay range, and the
model correctly predicts a value outside the assay range.
```bash
mkdir -p model_training/achiral_enantiopure_semiquant_schnet
python /path/to/ml/scripts/train.py \
    -i './docking_results/*/*_bound.pdb' \
    -exp cdd_achiral_enantiopure_semiquant_schema.json \
    -cache ./achiral_enantiopure_semiquant_struct_ds.pkl \
    -model_o ./model_training/achiral_enantiopure_semiquant_schnet/ \
    -model schnet \
    -n_epochs 300 \
    --wandb \
    -proj temporal-splits \
    --temporal \
    -pred_r pic50 \
    -sub 0.375 \
    -km 9.5 \
    -loss step
```
