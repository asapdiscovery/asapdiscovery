import abc
from typing import Literal, Tuple

from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.complex import Complex

from pydantic import BaseModel, Field


class LigandSelectorBase(abc.ABC, BaseModel):
    selector_type: Literal["LigandSelectorBase"] = Field(
        "LigandSelectorBase", description="The type of selector to use"
    )

    @abc.abstractmethod
    def _select(self, *args, **kwargs) -> list[Tuple[Ligand, Complex]]:
        ...

    def select(self, *args, **kwargs) -> list[Tuple[Ligand, Complex]]:
        return self._select(*args, **kwargs)

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]:
        ...
