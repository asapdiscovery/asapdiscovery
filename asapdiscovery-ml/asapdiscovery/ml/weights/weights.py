import logging
from pathlib import Path
from typing import Dict, List, Union  # noqa: F401

import requests
import validators
import yaml


def fetch_weights_from_spec(
    yamlfile: str,
    models: Union[list[str], str],
    local_dir: str = "./_weights/",
    force_fetch: bool = False,
) -> dict[str, Path]:
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
    Dict of model names and paths to weights files.
    """
    if not Path(yamlfile).exists():
        raise FileNotFoundError(f"Yaml spec file {yamlfile} does not exist")

    with open(yamlfile) as f:
        spec = yaml.safe_load(f)

    if isinstance(models, str):
        models = [models]

    weights_files = {}
    filename_set = set()
    for model in models:
        model_spec = spec[model]
        weights = model_spec["weights"]
        if weights in filename_set:
            raise ValueError(
                f"Duplicate file {weights} in spec file. Please specify a unique filename for each model."
            )
        weights_files[model] = fetch_weights(weights, local_dir, force_fetch)
        filename_set.add(weights)
    return weights_files


def fetch_weights(
    weights: str, file_dir: str = "./_weights/", force_fetch: bool = False
) -> Path:
    """Fetch weights from remote and save to local path.

    Parameters
    ----------
    weights : str
        Remote url or local path to weights.
    file_dir : str, default="./_weights/"
        Local path to save weights if a remote url is provided, or to check if weights exist locally, by default "./_weights/"
    force_fetch : bool, default=False
        Force fetch weights from remote, by default False

    Returns
    -------
    Path to weights file.
    """

    is_url = validators.url(weights)

    if not file_dir:
        raise ValueError("file_dir must be provided and must not be falsy.")

    if is_url:
        logging.info("Weights are a remote url")
        logging.info(f"Fetching weights from {weights}")
        if force_fetch:
            logging.info("Force fetching weights from remote")
            if not Path(file_dir).exists():
                logging.info(f"Local path {file_dir} does not exist, creating")
                Path(file_dir).mkdir(parents=True, exist_ok=True)
            # fetch from remote
            weights_file = download_file(weights, file_dir)
            return weights_file
        else:
            # check if weights exist locally
            local_filename = weights.split("/")[-1]
            local_path = Path(file_dir).joinpath(Path(local_filename))
            if local_path.exists():
                # exists locally, skip fetch from remote
                logging.info(
                    f"weights exist locally at {local_path}, skipping fetch from remote"
                )
                return Path(local_path)
            else:
                # fetch from remote
                logging.info("weights do not exist locally, fetching from remote")
                if not Path(file_dir).exists():
                    logging.info(f"Local path {file_dir} does not exist, creating")
                    Path(file_dir).mkdir(parents=True, exist_ok=True)
                weights_file = download_file(weights, file_dir)
                return weights_file
    else:
        local_path = Path(file_dir).joinpath(Path(weights))

        logging.info(f"Fetching weights from {local_path}")
        if not Path(local_path).exists():
            raise FileNotFoundError(
                f"Local weights file {local_path} does not exist, provide a valid remote url or local path to weights"
            )

        else:
            logging.info(
                f"weights exist locally at {local_path}, skipping fetch from remote"
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
