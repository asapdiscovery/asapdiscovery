import pathlib
from typing import List, Union  # noqa: F401

import pkg_resources
import pooch
import yaml

"""
This file contains utilities for fetching test files from the asapdiscovery
test file repository. We instantiate a pooch repository for the test files on import
that can then be used to fetch test files.
"""

test_files = pkg_resources.resource_filename(__name__, "test_files.yaml")


def make_test_file_pooch_repo(test_files: str) -> pooch.Pooch:
    """
    Make a pooch repository for test files.

    Parameters
    ----------
    test_files : str
        Path to yaml spec file.

    Returns
    -------
    pooch.Pooch
        Pooch repository for test files.

    Raises
    ------
    ValueError
        If test_files spec is invalid.
    """
    if not pathlib.Path(test_files).exists():
        raise ValueError(f"Spec file {test_files} does not exist.")
    with open(test_files) as f:
        test_files = yaml.safe_load(f)

    if "base_url" not in test_files:
        raise ValueError("base_url not found in spec file.")
    if "files" not in test_files:
        raise ValueError("files not found in spec file.")

    # make the registry
    reg = {}
    for fn in test_files["files"]:
        if "resource" not in fn:
            raise ValueError(f"File {fn} resource not found in spec file.")
        if "sha256hash" not in fn:
            raise ValueError(f"File {fn} sha256hash not found in spec file.")
        filename = fn["resource"]
        sha256hash = fn["sha256hash"]
        reg[filename] = f"sha256:{sha256hash}"

    return pooch.create(
        # use os cache to avoid permission issues and in-tree shenanigans
        path=pooch.os_cache("asapdiscovery_testing"),
        base_url=test_files["base_url"],
        registry=reg,
    )


# instantiate the pooch repository
test_file_pooch_repo = make_test_file_pooch_repo(test_files)


def fetch_test_file(filenames: Union[str, list[str]]) -> pathlib.Path:
    """
    Fetch a test file from the test file pooch repository.

    Parameters
    ----------
    filenames : Union[str, List[str]]
        Name of the test file or files to fetch.

    Returns
    -------
    pathlib.Path
        Path to the test file.

    Raises
    ------
    ValueError
        If the test file could not be fetched.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    files = []
    for filename in filenames:
        try:
            file = test_file_pooch_repo.fetch(filename)
        except Exception as e:
            raise ValueError(
                f"Could not fetch test file {filename} from {test_files}"
            ) from e
        files.append(pathlib.Path(file))
    if len(files) == 1:
        return files[0]
    else:
        return files
