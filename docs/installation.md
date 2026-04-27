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

`asapdiscovery` is a namespace package split across 11 subpackages, each independently installable with its own conda environment. Development uses [`just`](https://github.com/casey/just) as a task runner over this layout. Install it first if you don't have it (e.g. `mamba install -c conda-forge just`).

Clone the repository:

```bash
git clone git@github.com:asapdiscovery/asapdiscovery.git
cd asapdiscovery
```

Create a conda environment for the subpackage(s) you want to work on. Per-subpackage environment files live under `devtools/conda-envs/<platform>/`, plus an `all` environment that covers everything. The `just create-env` recipe picks the right platform automatically:

```bash
# Environment with every subpackage's dependencies
just create-env all asapdiscovery

# Or for a single subpackage (e.g. data)
just create-env data asapdiscovery-data

mamba activate asapdiscovery   # or your chosen env name
```

Install the subpackages in editable mode. To install everything:

```bash
just install-all
```

To install a single subpackage along with its internal dependencies in topological order:

```bash
just install-with-deps alchemy
```

To inspect the internal dependency graph:

```bash
just deps
```

Run tests for a single subpackage with `just test <pkg>`, or run them all sequentially with `just test-all`. Apply pre-commit linters across the repo with `just lint`. Run `just` with no arguments to list every available recipe.
