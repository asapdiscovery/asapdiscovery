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
download-moonshot-data \
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
cdd-to-schema \
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
download-fragalysis-data \
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
find-all-mcs \
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
This script takes advantage of the new `click` CLI, so it can only be called using the
CLI command `asap-prep`. In the future, the other steps in this pipeline will be updated
to follow this same pattern.

This example will prep every structure downloaded in Fragalysis
(`--fragalysis-dir ./mpro_fragalysis/`), and will store each individual prepped
structure as both a `.oedu` file and a JSON file
(`--gen-cache structures --cache-type DesignUnit --cache-type JSON`). We also use `dask`
to parallelize the job locally (`--use-dask --dask-type local`). We will also use the
RCSB OpenEye loop database (`--loop-db ./rcsb_spruce.loop_db`), which will need to be
downloaded separately.
```bash
mkdir -p prepped_receptors
asap-prep protein-prep \
    --target SARS-CoV-2-Mpro \
    --loop-db ./rcsb_spruce.loop_db \
    --fragalysis-dir ./mpro_fragalysis/ \
    --gen-cache structures \
    --cache-type DesignUnit \
    --cache-type JSON \
    --use-dask \
    --dask-type local \
    --output-dir ./prepped_receptors
```

## Run Docking
The final step in preparing the structural data is to run docking. There are a number of
options that can be configured, but this a minimal example that will dock each compound
to its top match from the MCS search. This step is again parallelized (`-n 32`), which
can be adjusted based on your machine. Note that the single quotes in the `-r` option
are required to prevent the shell from expanding the wildcard.
```bash
mkdir -p docking_results
run-docking-oe \
    -e cdd_achiral_enantiopure_semiquant_schema.json \
    -r './prepped_receptors/structures/*.oedu' \
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
make-docked-complexes-v2 \
    -d ./docking_results/ \
    -x ./mpro_fragalysis/ \
    -w 12
```

# Machine Learning
Finally we can get into the ML. As of PR
[#601](https://github.com/choderalab/asapdiscovery/pull/601/), the ML part of this
pipeline has drastically changed to be (hopefully) way easier to use. In this guide
I'll try to show how things can be done both with the new CLI and with the API.

## Some preliminary API notes
The API has been updated to follow the Pydantic schema pattern that the rest of the repo
has adopted. There are now 8 `*Config` classes (5 of which are implemented in
`asapdiscovery.ml.schema_v2.config` and 3 of which are implemented in `mtenn.config`).
Each config class is fairly well documented so I'll just give a brief summary here:
* `asapdiscovery.ml.schema_v2.OptimizerConfig`: A Config class describing the
optimizer to be used in training.
* `asapdiscovery.ml.schema_v2.EarlyStoppingConfig`: A Config class describing the
early stopping method to be used in training.
* `asapdiscovery.ml.schema_v2.DatasetConfig`: A Config class describing a Dataset
(either graph-based or structural). This Config has two convenient constructor
methods, one for constructing a graph-based Config from an experimental data file
(`DatasetConfig.from_exp_file`) and one for constructing a structural Config from
PDB file(s) and an optional experimental data file (`DatasetConfig.from_str_files`).
* `asapdiscovery.ml.schema_v2.DatasetSplitterConfig`: A Config class describing how
to split a Dataset. This Config differs from the others in that it does not build a
secondary object, but can instead be used directly.
* `asapdiscovery.ml.schema_v2.LossFunctionConfig`: A Config class describing a loss
function to be used in training.
* `mtenn.config.[GAT/SchNet/E3NN]ModelConfig`: Config classes describing an `mtenn`
GAT/SchNet/E3NN model. Each of these Configs also contains the parameters in the
abstract `mtenn.config.ModelConfigBase`, which are `mtenn`-specific parameters that
are common to all models.

All of these Config classes (with the exception of `DatasetSplitterConfig`) have a
`build` function, which uses the Config to create an instance of the described
object. These functions do not modify the Config, and return the created object, eg
`model = model_config.build()`.

## Prepare ML Dataset
Before training the model, we will build the input `DockedDataset` object and cache it
(and its corresponding `DatasetConfig` schema).
This step isn't strictly necessary, but it will save time during training and makes it
so that mulitple different training runs can be dispatched concurrently. Note that the
single quotes in the `--structures` option are required to prevent the shell from
expanding the wildcard. Note that a seperate dataset will need to be generated for
graph-based and e3nn models, but this same script call can be used by simply changing the option after `build-dataset` (eg `asap-ml build-dataset gat`).
```bash
asap-ml build-dataset schnet \
    --exp-file cdd_achiral_enantiopure_semiquant_schema.json \
    --structures './docking_results/*/*_bound.pdb' \
    --ds-cache ./achiral_enantiopure_semiquant_schnet_ds.pkl \
    --ds-config-cache ./achiral_enantiopure_semiquant_schnet_config.json
```
The equivalent Python code would be:
```python
from pathlib import Path

from asapdiscovery.data.utils import MOONSHOT_CDD_ID_REGEX, MPRO_ID_REGEX
from asapdiscovery.ml.schema_v2.config import DatasetConfig

ds_config = DatasetConfig.from_str_files(
    structures="./docking_results/*/*_bound.pdb",
    xtal_regex=MPRO_ID_REGEX,
    cpd_regex=MOONSHOT_CDD_ID_REGEX,
    for_training=True,
    exp_file="cdd_achiral_enantiopure_semiquant_schema.json",
    cache_file="./achiral_enantiopure_semiquant_schnet_ds.pkl",
)
Path("./achiral_enantiopure_semiquant_schnet_config.json").write_text(ds_config.json())
ds = ds_config.build()
```
After either of the above is run, the JSON file can then be loaded and the Dataset
object built (by loading from the cache) with:
```python
ds_config = DatasetConfig(
    **Path("cdd_achiral_enantiopure_semiquant_schema.json").read_text()
)
ds = ds_config.build()
```

## Train the Model
There are many different options for how this script can be called. This example shows
how to train a SchNet model using all the data previously generated in this example. We
will use the previously generated DatasetConfig and pickled Dataset object
(`--ds-config-cache ./achiral_enantiopure_semiquant_schnet_config.json`). We will train
for 300 epochs (`--n-epochs 300`) on the GPU (`--device cuda`), and use temporal
splitting for our data (`--ds-split-type temporal`). Additionally, this example uses
Weights & Biases to track
model training (`--use-wandb True`). You should set up and initialize your W&B account
according to their docs before running this. We will log this to a `test` project
(`--wandb-project test`), but you can obviously call this whatever you like. We will use
a `PIC50Readout` block in the model (`--pred-readout pic50`) to allow training against
experimental pIC50 values. Here we use the experimental condition values from the Moonshot
experiments (`--pred-substrate 0.375 --pred-km 9.5`), but these should be adjusted if
you're using different data. Finally, we use a step MSE loss function
(`--loss-type mse_step`), which prevents the model from being penalized if the data it's
trying to predict is outside the assay range, and the model correctly predicts a value
outside the assay range.

This command will also generate a JSON cache of the generated `Trainer` object, which
can then be loaded directly in any subsequent training runs. This can also be
accomplished by running `asap-ml build` instead of `asap-ml build-and-train`, with all
other options remaining the same. The `build` mode just builds and saves the `Trainer`
schema, without doing any initializing. The `build-and-train` mode, as the name implies,
instead builds/loads the `Trainer` schema, then initializes it and runs training.
```bash
asap-ml build-and-train schnet \
    --output-dir ./model_training/achiral_enantiopure_semiquant_schnet/ \
    --trainer-config-cache ./model_training/achiral_enantiopure_semiquant_schnet/trainer.json \
    --ds-split-type temporal \
    --ds-config-cache ./achiral_enantiopure_semiquant_schnet_config.json \
    --loss-type mse_step \
    --pred-readout pic50 \
    --pred-substrate 0.375 \
    --pred-km 9.5 \
    --device cuda \
    --n-epochs 300 \
    --use-wandb True \
    --wandb-project test
```
This can also be accomplished directly in Python, with:
