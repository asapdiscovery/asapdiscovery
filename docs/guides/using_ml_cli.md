Using the ML CLI
================

This guide will roughly follow the same flow as the corresponding [tutorial notebook](examples/training_ml_models_on_asap_data.ipynb).

% TODO: Switch these scripts to click instead of argparse and hook into main CLI

## Preparing the experimental data

Before using the data in training, we will do some filtering and processing of the experimental data to ensure that everything is in the correct format.
We will use all the default values for column names, which come from the [COVID Moonshot project](https://www.science.org/doi/10.1126/science.abo7201).
We'll start from an untiltered CSV downloaded from CDD, although this script can also be used to pull files from a saved search in a CDD vault.

The current iteration of this script requires a CDD API token, even if using a local CSV file, so we'll just set the env variable as an empty string.
We will pass our previously downloaded CSV file as the cache file, to avoid trying to download anything.
We will ultimately use this data to train both 2D and structure-based models, so we will keep all achiral and enantiopure molecules, including any molecules with semiquantitative fluorescence values.
This example assumes the input data columns are those from the Moonshot data, but this can be adjusted with the `--smiles_fieldname` and `--id_fieldname` CLI args.

The default behavior of this script is to use the Cheng-Prusoff equation to approximate the pIC50 values, using the values from the experimental conditions of the Moonshot fluorescence experiments.
To prevent this, you can pass `--cheng_prusoff 0 0` to disable using the Cheng-Prusoff equation.
If you'd like to pas your own values for this equation, see the help text of the CLI arg for more details.

```
CDDTOKEN="" download-cdd-data -o cdd_filtered_processed.csv -cache cdd_unfiltered.csv --retain_achiral --retain_enantiopure --retain_semiquant
```

The output of this script call should be the CSV file `cdd_filtered_processed.csv`, containing the filtered and processed data from CDD.
The last step in this process is to convert this data into the format that the ML pipeline expects it in.
The next script call does that, taking the previously generated CSV file as input and producing a JSON file that we will load later.

```
cdd-to-schema -i cdd_filtered_processed.csv -json cdd_filtered_processed.json
```

## Building the ML dataset

In this step we take our formatted experimental data and our docked complex structure PDB files and build the dataset objects that will be used during training.
This step can be folded into the same CLI call with the training, but we show it separately for ilustration purposes.
Having a separately generated daataset file that can be accessed by different models can save a lot of time in eg hyperparameter tuning.

For the 2D dataset, we only need to pass values for the path to the experimental data file and paths to where we want to cache the config and the dataset itself.
For the structural datasets, we pass these args as well as a glob describing the complex structure PDB files.
The structure datasets also take args for `--xtal-regex` and `--cpd-regex`, which are regular expressions describing how to extract the crystal structure and compound IDs respectively from each PDB file path.
These default to regexes for the Moonshot data, but if you're using your own data you'll need to provide values for these CLI args.

We assume that the docked PDB files are all in the top level of the directory `./docked_results/`, but this should be adjusted if that's not the case.

```
# Build GAT dataset
asap-ml build-ds gat --exp-file cdd_filtered_processed.json --ds-cache gat_ds.pkl --ds-config-cache gat_config.json

# Build SchNet dataset
asap-ml build-ds schnet --exp-file cdd_filtered_processed.json --ds-cache schnet_ds.pkl --ds-config-cache schnet_config.json --structures ./docked_results/*.pdb

# Build e3nn dataset
asap-ml build-ds e3nn --exp-file cdd_filtered_processed.json --ds-cache e3nn_ds.pkl --ds-config-cache e3nn_config.json --structures ./docked_results/*.pdb
```

The output of these commands should be a `<model>_config.json` and a `<model>_ds.pkl` file for each architecture.
These files can now be passed to the next step without having to reprocess all the data and structure files.

## Training the models

There are many CLI args that can be passed for this, defining every different part of training.
We'll leave most of them as the default here, but you can explore the different options by running `asap-ml build-and-train --help`.

We will be using the default Adam optimizer, default model hyperparameters with a pIC50 readout for the structure-basd models, no early stopping, temporal data splitting with an 80:10:10 split, and a semi-quantitative MSE loss function.
We will train for 500 epochs, with a mini-batch size of 25, on the GPU, and will save the model outputs to `<model>_training/.
We will log each training run to W&B, in a project named tutorial as a run named after the model, although this is optional`

```
# Train GAT model
asap-ml build-and-train gat --output-dir ./gat_training/ --trainer-config-cache ./gat_training/trainer.json --ds-split-type temporal --ds-cache gat_ds.pkl --ds-config-cache gat_config.json --loss-type mse_step --device cuda --n-epochs 500 --use-wandb True --wandb-project tutorial --wandb-name gat

# Train SchNet model
asap-ml build-and-train schnet --output-dir ./schnet_training/ --trainer-config-cache ./schnet_training/trainer.json --ds-split-type temporal --ds-cache schnet_ds.pkl --ds-config-cache schnet_config.json --loss-type mse_step --device cuda --n-epochs 500 --use-wandb True --wandb-project tutorial --wandb-name schnet

# Train e3nn model
asap-ml build-and-train e3nn --output-dir ./e3nn_training/ --trainer-config-cache ./e3nn_training/trainer.json --ds-split-type temporal --ds-cache e3nn_ds.pkl --ds-config-cache e3nn_config.json --loss-type mse_step --device cuda --n-epochs 500 --use-wandb True --wandb-project tutorial --wandb-name e3nn
```

In the above CLI calls, we passed a value for `--trainer-config-cache`.
This option serializes the `Trainer` object to a JSON file, so that it can be reused in future runs without needing to pass every single arg.
This functionality also helps with reproducibility, as there's no need to remember every single CLI arg you specified since it's all stored in the `trainer.json` file.
To generate these files without also training the model, you can replace the above CLI call with `asap-ml build <model>`, while keeping all the args the same.
This will generate the `trainer.json` file and then exit, rather than also building and training the model.

Once the above is run, future training runs can be carried out by simply passing the `trainer.json` file.

```
# Train GAT model
asap-ml build-and-train gat --trainer-config-cache ./gat_training/trainer.json

# Train SchNet model
asap-ml build-and-train schnet --trainer-config-cache ./schnet_training/trainer.json

# Train e3nn model
asap-ml build-and-train e3nn --trainer-config-cache ./e3nn_training/trainer.json
```
