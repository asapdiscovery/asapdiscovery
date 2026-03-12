# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**asapdiscovery** is a structure-based open antiviral drug discovery toolkit by the [ASAP Discovery Consortium](https://asapdiscovery.org/). It is a PEP 420 namespace package composed of 11 independently installable subpackages sharing the `asapdiscovery` namespace.

Python compatibility: **>=3.12, <3.14**

## Build & Development Setup

Dependencies are managed via conda/mamba (not pip). Create the dev environment:
```bash
mamba env create -f devtools/conda-envs/asapdiscovery-{platform}.yml
conda activate asapdiscovery
```

Install individual subpackages in editable mode (no-deps since conda provides dependencies):
```bash
pip install --no-deps -e ./asapdiscovery-data
pip install --no-deps -e ./asapdiscovery-docking
# ... etc for each needed package
```

## Testing

Tests use **pytest** across all packages. Each subpackage has its own test suite:
```bash
# Run tests for a single package
pytest asapdiscovery-data/asapdiscovery/data/tests/
pytest asapdiscovery-docking/asapdiscovery/docking/tests/

# Run a single test file
pytest asapdiscovery-data/asapdiscovery/data/tests/test_ligand.py

# Run a single test
pytest asapdiscovery-data/asapdiscovery/data/tests/test_ligand.py::test_function_name
```

Plugins in use: pytest-xdist (parallel), pytest-cov (coverage), pytest-timeout.

## Linting & Formatting

Pre-commit hooks handle formatting (run automatically in CI via pre-commit.ci):
- **black** (code formatting)
- **isort** (import sorting, profile: black)
- **flake8** (linting, max-line-length: 88, ignores E203/E501)
- **pyupgrade** (--py39-plus)

Run locally:
```bash
pre-commit run --all-files
```

## Architecture

### Package Dependency Graph

```
asapdiscovery-cli (aggregator)
  ├── asapdiscovery-workflows ──→ asapdiscovery-docking ──┐
  ├── asapdiscovery-alchemy                                ├──→ asapdiscovery-data (foundation)
  ├── asapdiscovery-ml ────────────────────────────────────┤
  ├── asapdiscovery-spectrum ──→ asapdiscovery-modeling ───┤
  ├── asapdiscovery-dataviz ───────────────────────────────┤
  └── asapdiscovery-simulation ────────────────────────────┘
```

**asapdiscovery-data** is the foundation layer. All other packages depend on its schemas, backends, and utilities.

### Core Design Patterns

**Schema-first with Pydantic v1:** All data models inherit from `DataModelAbstractBase` (in `asapdiscovery-data/.../schema/schema_base.py`), which provides JSON serialization, hashing, equality checks, and `from_json_file()`/`to_json_file()` methods. Key models: `Ligand`, `Target`, `Complex`, `PreppedComplex`, with associated `*Identifiers` classes.

**Backend abstraction:** Chemistry toolkit operations are wrapped in `asapdiscovery.data.backend.openeye` and `asapdiscovery.data.backend.rdkit`, providing a consistent interface regardless of the underlying toolkit.

**Factory pattern for data loading:** `MetaStructureFactory`, `MetaLigandFactory`, `StructureDirFactory`, `MolFileFactory` in `asapdiscovery.data.readers` abstract over different data sources (directories, Fragalysis, PDB, etc.).

**Dask integration:** Distributed computing support via `dask_vmap` and `BackendType`/`FailureMode` enums in `asapdiscovery.data.util.dask_utils`.

### CLI Structure

Click-based CLI with a hub-and-spoke pattern:
- Main entry point: `asap-cli` → `asapdiscovery.cli.cli:cli` (dynamically adds subcommands from all packages)
- Delegated CLIs: `asap-docking`, `asap-prep`, `asap-alchemy`, `asap-ml`, `asap-spectrum`
- Data utilities: `cdd-to-schema`, `download-cdd-data`, `download-pdbs`, `split-fragment-screen`

### Key External Dependencies

- **OpenEye toolkits** — primary chemistry engine (requires license)
- **RDKit** — secondary cheminformatics
- **OpenMM / mdtraj** — molecular dynamics
- **PyTorch / PyTorch Geometric / DGL / MTENN** — ML infrastructure
- **OpenFE / Alchemiscale** — free energy calculations
- **Pydantic v1** (pinned `>=1.10.8,<2.0.0a0`)

### Versioning

Uses **versioningit** — versions are derived from git tags automatically. No manual version bumps needed.
