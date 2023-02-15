asapdiscovery
=============
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/choderalalb/asapdiscovery/workflows/CI/badge.svg)](https://github.com/choderalab/asapdiscovery/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/choderalab/asapdiscovery/branch/main/graph/badge.svg)](https://codecov.io/gh/choderalab/asapdiscovery/branch/main)


Scripts and models for ML with COVID Moonshot data.


## How to use

### Intro
This is a brief depiction of an example run through the pipeline in the [`asapdiscovery` repo](https://github.com/choderalab/asapdiscovery). All Python scripts mentioned in this tutorial are located in the top-level `scripts/` directory in the repo. I will also use the variable `CML_DIR` as the directory where the repo has been downloaded. The steps in the pipeline are:
1. Download Fragalysis and COVID Moonshot data
2. Parse data into correct format for ML
3. Run the docking pipeline
4. ML
### Downloading the data
The steps in this section are independent and can be run concurrently.
#### Download Fragalysis Data
**Script:** `download_fragalysis_data.py`

**Options:**
- `-o` (required): The full file name of the `.zip` file that will be downloaded.
- `-x` (optional flag): If present, the script will unzip the archive after it is downloaded.

**Example usage:** The following code will download the full Fragalysis archive and unpack all the files.
```bash
mkdir fragalysis/
python ${CML_DIR}/scripts/download_fragalysis_data.py -o ./fragalysis/fragalysis.zip -x
```

If this code executes successully, you will be left with a `fragalysis/` directory containing the `fragalysis.zip` file as well as its contents.
#### Download Moonshot Data
**Script:** `download_moonshot_data.py`

**Options:**
- `-tok` (required): A file containing the user's CDD vault token.
- `-o` (required): The full file name of the CSV file to output.

**Example usage:** The following code will download the achiral data in the Moonshot CDD vault as a CSV file. This assumes that the user has a valid CDD vault token stored in plain text as `./cdd_vault_token.txt`.
```bash
python ${CML_DIR}/scripts/download_moonshot_data.py -tok ./cdd_vault_token.txt -o ./cdd_moonshot_achiral.csv
```

### Parse data into correct format
The steps in this section are independent and can be run concurrently.
#### Parse Fragalysis Data
As of the time of writing, the PDB files downloaded from Fragalysis do not include the `SEQRES` header, so we need to add it.

**Script:** `mass_add_seqres.py`

**Options:**
- `-i` (required):  Fragalysis sub-directory containing the aligned Mpro structures (`fragalysis/aligned/`)
**Example usage:** This example continues from the downloading example in the previous step.
```bash
python ${CML_DIR}/scripts/mass_add_seqres.py -i ./fragalysis/aligned/
```

#### Parse Moonshot Data
The data gets downloaded from the CDD vault in a CSV file, so we need to parse it into the `Pydantic` schema objects.

**Script:** `cdd_to_schema.py`

**Options:**
- `-i` (required): CSV input file downloaded from the CDD vault.
- `-o` (required): Full output file path of the parsed JSON file.
- `-type` (optional [std, ep], default=std): What type of data is being loaded.
  - `std`: Standard data, each compound is treated separately.
  - `ep`: Enantiomer pairs, compounds are treated as stereochemically pure enantiomer pairs.
- `-achiral` (optional flag): If present, the script will only keep achiral molecules.

**Example usage:** This example continues from the downloading example in the previous step. It will parse the file as a list of separate compounds, filtering out achiral molecules.
```bash
python ${CML_DIR}/scripts/cdd_to_schema.py -i ./cdd_moonshot_achiral.csv -o ./cdd_moonshot_achiral.json -achiral
```

### Run the docking pipeline
The steps in this section are **not** independent and must be run in the order they're listed.
#### Running the MCS search
Run an MCS search between all experimental compounds and all crystal structure compounds.

**Script:** `find_all_mcs.py`

**Options:**
- `-exp` (required): JSON file with experimental results in schema format.
- `-x` (required): CSV file from Fragalysis with information on the crystal structures.
- `-x_dir` (required): Fragalysis `aligned/` directory with complex crystal structures.
- `-o` (requred): Top-level output directory.
- `-n` (int, default=1): Number of concurrent processes to run.
- `-sys` (optional [rdkit, oe], default=rdkit): Which package to use for MCS search.
- `-str` (optional flag): If present, the script will perform a structure-based MCS search instead of requiring matching elements.
- `-ep` (optional flag): If present, indicates that the input data is a list of enantiomer pairs instead of individual compounds.

**Example usage:** Perform element-based MCS search using `rdkit` for the downloaded/parsed files from before, using 16 concurrent processes.
```bash
mkdir mcs_res/
python ${CML_DIR}/scripts/find_all_mcs.py \
-exp ./cdd_moonshot_achiral.json \
-x ./fragalysis/extra_files/Mpro_compound_tracker_csv.csv \
-x_dir ./fragalysis/aligned/ \
-o ./mcs_res/ \
-n 16
```

#### Running docking
Run the docking process based on the MCS results.

**Script:** `run_docking.py`

**Options:**
- `-exp` (required): JSON file with experimental results in schema format.
- `-x` (required): CSV file from Fragalysis with information on the crystal structures.
- `-x_dir` (required): Fragalysis `aligned/` directory with complex crystal structures.
- `-loop` (required): Spruce loop database file.
- `-mcs` (required): Pickle file with the MCS search results.
- `-o` (required): Top-level output directory.
- `-cache` (optional): Docking cache directory. Will use `.cach` in the output directory if not given.
- `-n` (int, default=1): Number of concurrent processes to run.
- `-achiral` (optional flag): If present, the script will only keep achiral molecules.
- `-ep` (optional flag): If present, indicates that the input data is a list of enantiomer pairs instead of individual compounds.

**Example usage:** Run docking using results from previous step. Use 16 concurrent processes, filter for achiral molecules, use default cache directory. Assumes loop database is present in the current directory as `rcsb_spruce.loop_db`.
```bash
mkdir docking_res/

##TODO: THIS SCRIPT HAS SINCE BEEN DELETED, need to rewrite this documentation
python ${CML_DIR}/scripts/run_docking.py \
-exp ./cdd_moonshot_achiral.json \
-x ./fragalysis/extra_files/Mpro_compound_tracker_csv.csv \
-x_dir ./fragalysis/aligned/ \
-loop rcsb_spruce.loop_db \
-mcs ./mcs_res/mcs_sort_index.pkl \
-o ./docking_res/ \
-n 16 \
-achiral
```

### ML
Train the model with the docked structures.

**Script:** `train_dd.py`

### Copyright

Copyright (c) 2022, kaminow


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
