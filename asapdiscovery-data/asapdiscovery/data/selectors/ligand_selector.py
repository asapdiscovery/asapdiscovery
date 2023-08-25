import abc
from typing import Literal

from asapdiscovery.data.schema_v2.complex import Complex
from asapdiscovery.data.schema_v2.ligand import Ligand
from pydantic import BaseModel, Field


class LigandSelectorBase(abc.ABC, BaseModel):
    # records what kind of selector class was used, overridden in subclasses
    selector_type: Literal["LigandSelectorBase"] = Field(
        "LigandSelectorBase", description="The type of selector to use"
    )

    @abc.abstractmethod
    def _select(self, *args, **kwargs) -> list[tuple[Ligand, Complex]]:
        ...

    def select(self, *args, **kwargs) -> list[tuple[Ligand, Complex]]:
        return self._select(*args, **kwargs)

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]:
        ...
