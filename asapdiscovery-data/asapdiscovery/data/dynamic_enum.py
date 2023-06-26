from enum import Enum
from pathlib import Path

import yaml


class DynamicEnum(Enum):
    ...


def make_dynamic_enum(yaml_path: str | Path, enum_name: str) -> DynamicEnum:
    """Make a dynamic enum from a yaml file.

    Args:
        yaml_path (str | Path): Path to yaml file

    Returns:
        Enum: Enum object
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return DynamicEnum(enum_name, data)
