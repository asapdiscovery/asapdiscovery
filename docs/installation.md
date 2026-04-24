Installation
===============

This page details how to get started with `asapdiscovery` and how to install it on your system.

There are two ways to install `asapdiscovery`:

1. From conda-forge (recommended)
2. Developer installation from source

Installation from conda-forge
----------------------------

The easiest way to install `asapdiscovery` is to use the mamba (or conda) package manager. You can install `asapdiscovery` from the `conda-forge` channel using the following command:
The openeye package is not available in the conda-forge channel, so you need to install it from the openeye channel. You will need to have an OpenEye license to use some functionality in the package.
You can request a free academic license from the [OpenEye website](https://docs.eyesopen.com/toolkits/python/index.html).

```bash
mamba create -n asapdiscovery python=3.10
mamba activate asapdiscovery
mamba install -c conda-forge asapdiscovery
mamba install -c openeye openeye-toolkits

```

Developer installation from source
----------------------------------

To install `asapdiscovery` from source, you will need to clone the repository, setup a compatible environment with mamba (or conda) and install the package using `pip`. You can do this using the following commands:

```bash
git clone git@github.com:asapdiscovery/asapdiscovery.git
cd asapdiscovery
mamba env create -f devtools/conda-envs/asapdiscovery-ubuntu-latest.yml # chose relevant file for your platform
mamba activate asapdiscovery
cp devtools/repo_installer.sh . && chmod +x repo_installer.sh && ./repo_installer.sh
```

This will install the package in editable mode, so you can make changes to the code and see the changes reflected in the package. You can also run the tests using the following command:
