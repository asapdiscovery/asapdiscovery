import yaml

from enum import Enum
from pathlib import Path


def make_dynamic_enum(yaml_path: str | Path, enum_name: str) -> Enum:
    """Make a dynamic enum from a yaml file.

    Args:
        yaml_path (str | Path): Path to yaml file

    Returns:
        Enum: Enum object
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    print(data)
    return Enum(enum_name, data)
