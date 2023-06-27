from enum import Enum
from pathlib import Path
from typing import Union

import yaml


def make_dynamic_enum(yaml_path: Union[str, Path], enum_name: str) -> Enum:
    """Make a dynamic enum from a yaml file.

    Parameters
    ----------
    yaml_path : Union[str, Path]
        Path to yaml file
    enum_name : str
        Name of enum

    Returns
    -------
    Enum
        Enum object
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return Enum(enum_name, data)
