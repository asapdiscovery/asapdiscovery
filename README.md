asapdiscovery
=============
[//]: # (Badges)
[![codecov](https://codecov.io/gh/choderalab/asapdiscovery/branch/main/graph/badge.svg)](https://codecov.io/gh/choderalab/asapdiscovery/branch/main)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/choderalab/asapdiscovery/main.svg)](https://results.pre-commit.ci/latest/github/choderalab/asapdiscovery/main)
[![Documentation Status](https://readthedocs.org/projects/asapdiscovery/badge/?version=latest)](https://asapdiscovery.readthedocs.io/en/latest/?badge=latest)


A toolkit for structure-based open antiviral drug discovery by the [ASAP Discovery Consortium](https://asapdiscovery.org/).

<img src="docs/_static/asap_logo.png" width="500">


## Intro

All pandemics are global health threats. Our best defense is a healthy global antiviral discovery community with a robust pipeline of open discovery tools. The AI-driven Structure-enabled Antiviral Platform (ASAP) is making this a reality!

The toolkit in this repo is a batteries-included drug discovery pipeline being actively developed in a transparent open-source way, with a focus on computational chemistry and informatics support for medicinal chemistry. Coupled with ASAP's [active data disclosures](https://asapdiscovery.org/outputs/) our campaign to develop a new series of antivirals can provide insight into the drug discovery process that is normally conducted behind closed doors.


## Getting Started

Install the `asapdiscovery` subpackages and begin to explore! Our [docs can be found here](https://asapdiscovery.readthedocs.io/en/latest).

There are a range of workflows and tooling to use split into several namespace subpackages by theme.

`asapdiscovery-alchemy`: Free energy calculations using [OpenFE](https://openfree.energy/) and [Alchemiscale](https://docs.alchemiscale.org/en/latest/). See [tutorial](https://asapdiscovery.readthedocs.io/en/latest/tutorials/running_alchemical_free_energy_calculations.html) and CLI [guide](https://asapdiscovery.readthedocs.io/en/latest/guides/using_asap_alchemy_cli.html)

`asapdiscovery-cli`: Command line tools uniting the whole repo.

`asapdiscovery-data`: Core data models and integrations with services such as [Postera.ai](https://postera.ai/). See [tutorial](https://asapdiscovery.readthedocs.io/en/latest/tutorials/interfacing_with_databases_and_systems.html)

`asapdiscovery-dataviz`: Data and structure visualization using `3DMol` and `PyMOL`. See [tutorial](https://asapdiscovery.readthedocs.io/en/latest/tutorials/visualizing_asap_targets.html)

`asapdiscovery-docking`: Docking and compound screening with the `OpenEye` toolkit

`asapdiscovery-genetics`: Working with sequence and fitness information. See [tutorial](https://asapdiscovery.readthedocs.io/en/latest/tutorials/exploring_related_sequences_and_structures.html)

`asapdiscovery-ml`: Structure based ML models for predicting compound activity. See [tutorial](https://asapdiscovery.readthedocs.io/en/latest/tutorials/training_ml_models_on_asap_data.html) and CLI [guide](https://asapdiscovery.readthedocs.io/en/latest/guides/using_ml_cli.html)

`asapdiscovery-modelling`: Structure prep and standardisation

`asapdiscovery-simulation`: MD simulations and analysis using OpenMM. See [tutorial](https://asapdiscovery.readthedocs.io/en/latest/tutorials/running_md_simulations.html)

`asapdiscovery-workflows`: Workflows that combine components to enable project support. See [tutorial](https://asapdiscovery.readthedocs.io/en/latest/tutorials/docking_and_scoring.html) and CLI [guide](https://asapdiscovery.readthedocs.io/en/latest/guides/using_docking_cli.html)


### Disclaimer

`asapdiscovery` is pre-alpha and is under very active development, we make no guarantees around correctness and the API is liable to change rapidly at any time.


## Installation

**Note**: currently all `asapdiscovery` packages support Python 3.10 only.

`asapdiscovery` is a namespace package, composed of individual Python packages with their own dependencies.
Each of these packages follows the `asapdiscovery-*` convention for the package name, e.g. `asapdiscovery-data`.

To install an `asapdiscovery` package hosted in this repository, we recommend the following:

1. Clone the repository, then enter the source tree:

    ```
    git clone https://github.com/choderalab/asapdiscovery.git
    cd asapdiscovery
    ```

2. Install the dependencies into a new `conda` environment, and activate it:
   NOTES: Conda will almost certainly fail to build the environment - `mamba` is a drop-in replacement for `conda` that is much faster and more reliable.  Additionally, if the environment is built on a CPU, `torch` may not compile with GPU support. Instead, build the environment as described on a GPU node; the architecture will be detected automatically. Alternatively to build for a specific CUDA version you can use the

    ```
    mamba env create -f devtools/conda-envs/asapdiscovery-{platform}.yml
    conda activate asapdiscovery
    ```
    Alternatively to build for a specific CUDA version you can use the following.
    ```
    export CONDA_OVERRIDE_CUDA=12.2 && mamba env create -f devtools/conda-envs/asapdiscovery-{platform}.yml && conda activate asapdiscovery
    ```


3. Install the individual `asapdiscovery` packages you want to use with `pip` into the environment.
   For example, `asapdiscovery-data`:

    ```
    pip install asapdiscovery-data
    ```


### Contributing

### [pre-commit](https://pre-commit.com/#intro)

We use pre-commit to automate code formatting and other fixes.
You do not need to install pre-commit as we run it on our CI.
If you want to run it locally:
```bash
# install
$ mamba install -c conda-forge pre-commit
# check
$ pre-commit --version
pre-commit 3.0.4 # your version may be different
$ pre-commit install
```

Now every time you make a commit, the hooks will run on just the files you changed.
See [here](https://pre-commit.com/#usage) for more details.

### Copyright

Copyright (c) 2023, ASAP Discovery


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
