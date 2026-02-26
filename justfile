# asapdiscovery task runner
# https://github.com/casey/just

packages := "alchemy cli data dataviz docking ml modeling simulation spectrum workflows"

# List available recipes
default:
    @just --list

# Create conda dev environment for a single module (e.g. just create-env alchemy myenv -- -y)
create-env module name *FLAGS:
    #!/usr/bin/env bash
    set -euo pipefail
    case "$(uname -s)-$(uname -m)" in
        Linux-*)       platform="ubuntu-latest" ;;
        Darwin-arm64)  platform="macos-latest" ;;
        Darwin-x86_64) platform="macos-latest" ;;
        *)             echo "Unsupported platform: $(uname -s)-$(uname -m)"; exit 1 ;;
    esac
    env_file="devtools/conda-envs/${platform}/{{ module }}.yaml"
    if [ ! -f "$env_file" ]; then
        echo "No env file found: $env_file"
        echo "Available modules: all alchemy cli data dataviz docking ml modeling simulation spectrum workflows"
        exit 1
    fi
    echo "Using env file: $env_file"
    micromamba create -n {{ name }} -f "$env_file" {{ FLAGS }}

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
