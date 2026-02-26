# asapdiscovery task runner
# https://github.com/casey/just

packages := "alchemy cli data dataviz docking ml modeling simulation spectrum workflows"

# List available recipes
default:
    @just --list

# Create conda dev environment (auto-detects platform)
create-env name:
    #!/usr/bin/env bash
    set -euo pipefail
    case "$(uname -s)-$(uname -m)" in
        Linux-*)       env_file="devtools/conda-envs/asapdiscovery-ubuntu-latest.yml" ;;
        Darwin-arm64)  env_file="devtools/conda-envs/asapdiscovery-macOS-latest.yml" ;;
        Darwin-x86_64) env_file="devtools/conda-envs/asapdiscovery-macOS-12.yml" ;;
        *)             echo "Unsupported platform: $(uname -s)-$(uname -m)"; exit 1 ;;
    esac
    echo "Using env file: $env_file"
    micromamba create -n {{ name }} -f "$env_file"

# Install a single subpackage (e.g. just install data)
install pkg:
    pip install --no-deps -e ./asapdiscovery-{{ pkg }}

# Install all subpackages in editable mode
install-all:
    #!/usr/bin/env bash
    set -euo pipefail
    for pkg in {{ packages }}; do
        echo "Installing asapdiscovery-$pkg..."
        pip install --no-deps -e "./asapdiscovery-$pkg"
    done

# Run tests for a single subpackage (e.g. just test data)
test pkg:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{ pkg }}" = "workflows" ]; then
        pytest asapdiscovery-{{ pkg }}/asapdiscovery/{{ pkg }}/tests/ \
            --durations=10 -v --cov-report=term --color=yes \
            --cov=asapdiscovery-{{ pkg }}
    else
        pytest asapdiscovery-{{ pkg }}/asapdiscovery/{{ pkg }}/tests/ \
            -n auto --durations=10 -v --cov-report=term --color=yes \
            --cov=asapdiscovery-{{ pkg }}
    fi

# Run tests for all subpackages sequentially
test-all:
    #!/usr/bin/env bash
    set -euo pipefail
    for pkg in {{ packages }}; do
        echo "=== Testing asapdiscovery-$pkg ==="
        just test "$pkg"
    done

# Run pre-commit linters on all files
lint:
    pre-commit run --all-files
