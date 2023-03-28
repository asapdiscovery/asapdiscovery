import logging
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple, Union  # noqa: F401

import requests
import validators
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

    specs = {}
    # cannot specify the same weights file for multiple models
    weights_set = set()
    for model in models:
        model_spec = spec[model]
        weights = model_spec["weights"]
        model_type = model_spec["type"]
        if "config" in model_spec:
            config = model_spec["config"]
            config = fetch_file(config, local_dir, force_fetch)
        else:
            config = None
        if weights in weights_set:
            raise ValueError(
                f"Duplicate file {weights} in spec file. Please specify a unique filename for each model."
            )
        weight_file = fetch_file(weights, local_dir, force_fetch)
        weights_set.add(weights)
        specs[model] = ModelSpec(model, model_type, weight_file, config)
    return specs


def fetch_file(
    filename: str, file_dir: str = "./_weights/", force_fetch: bool = False
) -> Path:
    """Fetch weights from remote and save to local path.

    Parameters
    ----------
    filename : str
        Remote url or local path to file.
    file_dir : str, default="./_weights/"
        Local path to save weights if a remote url is provided, or to check if weights exist locally, by default "./_weights/"
    force_fetch : bool, default=False
        Force fetch file from remote, by default False

    Returns
    -------
    Path to file.
    """

    is_url = validators.url(filename)

    if not file_dir:
        raise ValueError("file_dir must be provided and must not be falsy.")

    if is_url:
        logging.info("file is a remote url")
        logging.info(f"Fetching file from {filename}")
        if force_fetch:
            logging.info("Force fetching file from remote")
            # fetch from remote
            target_file = download_file(filename, file_dir)
            return target_file
        else:
            # check if file exists locally
            local_filename = filename.split("/")[-1]
            local_path = Path(file_dir).joinpath(Path(local_filename))
            if local_path.exists():
                # exists locally, skip fetch from remote
                logging.info(
                    f"filename exist locally at {local_path}, skipping fetch from remote"
                )
                return Path(local_path)
            else:
                # fetch from remote
                logging.info("filename does not exist locally, fetching from remote")
                target_file = download_file(filename, file_dir)
                return target_file
    else:
        local_path = Path(file_dir).joinpath(Path(filename))

        logging.info(f"Fetching file from {local_path}")
        if not Path(local_path).exists():
            raise FileNotFoundError(
                f"Local file {local_path} does not exist, provide a valid remote url or local path to file"
            )

        else:
            logging.info(
                f"file exists locally at {local_path}, skipping fetch from remote"
            )
            return Path(local_path)


# https://stackoverflow.com/questions/16694907/
def download_file(url: str, path: str) -> None:
    """Download file from url and save to path.

    Parameters
    ----------
    url : str
        Remote url to download file from.
    path : str
        Local path to save file to.
    """
    if not Path(path).exists():
        logging.info(f"Local path {path} does not exist, creating")
        Path(path).mkdir(parents=True, exist_ok=True)

    if Path(path).is_dir():
        local_filename = url.split("/")[-1]
        path = Path(path).joinpath(Path(local_filename))
    else:
        path = Path(path)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return path
