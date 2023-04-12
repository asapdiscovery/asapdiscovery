import yaml
import pooch

from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple, Union  # noqa: F401

import yaml

ModelSpec = namedtuple("ModelSpec", ["name", "type", "weights", "config"])


def check_spec_validity(model: str, spec: dict) -> None:
    """Check if model spec is valid.

    Parameters
    ----------
    model : str
        Model name.
    spec : dict
        Model spec.

    Raises
    ------
    ValueError
        If model spec is invalid.
    """
    if model not in spec:
        raise ValueError(f"Model {model} not found in spec file.")
    model_spec = spec[model]
    if "type" not in model_spec:
        raise ValueError(f"Model {model} type not found in spec file.")
    if "base_url" not in model_spec:
        raise ValueError(f"Model {model} base_url not found in spec file.")
    if "weights" not in model_spec:
        raise ValueError(f"Model {model} weights not found in spec file.")
    if "resource" not in model_spec["weights"]:
        raise ValueError(f"Model {model} weights resource not found in spec file.")
    if "sha256hash" not in model_spec["weights"]:
        raise ValueError(f"Model {model} weights sha256hash not found in spec file.")
    if "config" in model_spec:
        if "resource" not in model_spec["config"]:
            raise ValueError(f"Model {model} config resource not found in spec file.")
        if "sha256hash" not in model_spec["config"]:
            raise ValueError(f"Model {model} config sha256hash not found in spec file.")


def fetch_model_from_spec(
    yamlfile: str,
    models: Union[list[str], str],
    local_dir: str = "./_weights/",
    force_fetch: bool = False,
) -> dict[str, tuple[Path, Path, str]]:
    """Fetch weights from yaml spec file.

    Parameters
    ----------
    yamlfile : str
        Path to yaml spec file.
    models : List[str]
        Model names to fetch weights for.
    local_dir : str, default="./_weights/"
        Local path to save weights if a remote url is provided. or to check if weights exist locally, by default "./_weights/"
    force_fetch : bool, default=False
        Force fetch weights from remote, by default False

    Raises
    ------
    FileNotFoundError
        If YAML spec file does not exist.

    Returns
    -------
    Dict of model names and weights paths.
    """
    if not Path(yamlfile).exists():
        raise FileNotFoundError(f"Yaml spec file {yamlfile} does not exist")

    with open(yamlfile) as f:
        spec = yaml.safe_load(f)

    if isinstance(models, str):
        models = [models]

    if not local_dir:
        raise ValueError("local_dir must be provided and must not be falsy.")

    if not Path(local_dir).exists():
        Path(local_dir).mkdir(parents=True, exist_ok=True)

    specs = {}
    for model in models:
        check_spec_validity(model, spec)
        model_spec = spec[model]
        model_type = model_spec["type"]
        base_url = model_spec["base_url"]

        weights = model_spec["weights"]
        weights_resource = weights["resource"]
        weights_hash = weights["sha256hash"]

        registry = {}
        registry[weights_resource] = f"sha256:{weights_hash}"

        # fetch config if provided
        if "config" in model_spec:
            config = model_spec["config"]
            config_resource = config["resource"]
            config_hash = config["sha256hash"]
            registry[config_resource] = f"sha256:{config_hash}"
        else:
            config_resource = None

        # make pooch registry
        subdir = model
        registry = pooch.create(
            path=Path(local_dir).joinpath(Path(subdir)),
            base_url=base_url,
            registry=registry,
        )

        # fetch weights
        try:
            weights_file = Path(registry.fetch(weights_resource))
        except:
            raise ValueError(
                f"Model {model} weights file {weights_resource} download failed, please check your yaml spec file for errors."
            )
        # fetch config
        if config_resource:
            try:
                config_file = Path(registry.fetch(config_resource))
            except:
                raise ValueError(
                    f"Model {model} config file {config_resource} download failed, please check your yaml spec file for errors."
                )
        else:
            config_file = None
        if model in specs:
            raise ValueError(
                f"Model {model} already exists in specs, please check your yaml spec file for duplicates."
            )
        # make model spec
        specs[model] = ModelSpec(model, model_type, weights_file, config_file)

    return specs
