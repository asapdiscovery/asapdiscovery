from enum import Enum


class StringEnum(str, Enum):
    @classmethod
    def get_values(cls) -> list[str]:
        return [member.value for member in cls]

    @classmethod
    def reverse_lookup(cls, value):
        return cls(value)

    @classmethod
    def get_names(cls) -> list[str]:
        return [member.name for member in cls]
