import logging
from pathlib import Path
from typing import List, Union

import requests
import validators
import yaml


def fetch_weights_from_spec(
    yamlfile: str,
    models: Union[list[str], str],
    local_path: str = "./_weights/",
    force_fetch: bool = False,
):
    """Fetch weights from yaml spec file.

    Parameters
    ----------
    yamlfile : str
        Path to yaml spec file.
    models : List[str]
        Model names to fetch weights for.
    local_path : str, optional
        Local path to save weights if a remote url is provided. or to check if weights exist locally, by default "./_weights/"
    force_fetch : bool, optional
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
        weights_files[model] = fetch_weights(weights, local_path, force_fetch)
        if weights in filename_set:
            raise ValueError(
                f"Duplicate file {weights} in spec file. Please specify a unique filename for each model."
            )
        filename_set.add(weights)
    return weights_files


def fetch_weights(weights: str, path: str = None, force_fetch: bool = False) -> Path:
    """Fetch weights from remote and save to local path.

    Parameters
    ----------
    weights : str
        Remote url or local path to weights.
    path : str, optional
        Local path to save weights if a remote url is provided. or to check if weights exist locally, by default None
    force_fetch : bool, optional
        Force fetch weights from remote, by default False

    Returns
    -------
    Path to weights file.
    """

    is_url = validators.url(weights)

    if is_url:
        logging.info("Weights are a remote url")
        logging.info(f"Fetching weights from {weights}")
        if force_fetch:
            logging.info("Force fetching weights from remote")
            if not Path(path).exists():
                logging.info(f"Local path {path} does not exist, creating")
                Path(path).mkdir(parents=True, exist_ok=True)
            # fetch from remote
            weights_file = download_file(weights, path)
            return weights_file
        else:
            # check if weights exist locally
            local_filename = weights.split("/")[-1]
            if Path(path + local_filename).exists():
                # exists locally, skip fetch from remote
                logging.info(
                    f"weights exist locally at {path + local_filename}, skipping fetch from remote"
                )
                return Path(path + local_filename)
            else:
                # fetch from remote
                logging.info("weights do not exist locally, fetching from remote")
                if not Path(path).exists():
                    logging.info(f"Local path {path} does not exist, creating")
                    Path(path).mkdir(parents=True, exist_ok=True)
                weights_file = download_file(weights, path)
                return weights_file
    else:
        logging.info(f"Fetching weights from {path + weights}")
        if not Path(path + weights).exists():
            raise FileNotFoundError(
                f"Local weights file {path + weights} does not exist, provide a valid remote url or local path to weights"
            )

        else:
            logging.info(
                f"weights exist locally at {path + weights}, skipping fetch from remote"
            )
            return Path(path + weights)


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
    local_filename = url.split("/")[-1]
    path = Path(path + local_filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return path
