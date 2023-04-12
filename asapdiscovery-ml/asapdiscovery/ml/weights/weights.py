from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple, Union  # noqa: F401

import pooch
import yaml

ModelSpec = namedtuple("ModelSpec", ["name", "type", "weights", "config"])


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
        weights_file = Path(registry.fetch(weights_resource))
        # fetch config
        if config_resource:
            config_file = Path(registry.fetch(config_resource))
        else:
            config_file = None
        # make model spec
        specs[model] = ModelSpec(model, model_type, weights_file, config_file)

    return specs
