import pkg_resources

from ..dynamic_enum import make_dynamic_enum

target_type_yaml = pkg_resources.resource_filename(__name__, "target_type.yaml")

TargetType = make_dynamic_enum(target_type_yaml, "TargetType")
