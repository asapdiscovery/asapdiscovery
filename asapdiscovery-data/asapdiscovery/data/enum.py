from enum import Enum


class StringEnum(str, Enum):
    @classmethod
    def get_values(cls, underscored: bool = False) -> list[str]:
        if underscored:
            return [member.value.replace("-", "_") for member in cls]
        else:
            return [member.value for member in cls]

    @classmethod
    def reverse_lookup(cls, value):
        return cls(value)

    @classmethod
    def get_names(cls) -> list[str]:
        return [member.name for member in cls]
