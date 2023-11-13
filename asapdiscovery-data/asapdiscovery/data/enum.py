from enum import Enum


class StringEnum(str, enum):
    @classmethod
    def get_values(cls) -> list[str]:
        return [member.value for member in cls]

    @classmethod
    def reverse_lookup(cls, value) -> StringEnum:
        return cls(value)

    @classmethod
    def get_names(cls) -> list[str]:
        return [member.name for member in cls]
