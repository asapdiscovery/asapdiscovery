import abc
from typing import Literal, Union

from asapdiscovery.data.schema_v2.pairs import CompoundStructurePair, DockingInputPair

from pydantic import BaseModel, Field


class LigandSelectorBase(abc.ABC, BaseModel):
    # records what kind of selector class was used, overridden in subclasses
    selector_type: Literal["LigandSelectorBase"] = Field(
        "LigandSelectorBase", description="The type of selector to use"
    )

    @abc.abstractmethod
    def _select(self, *args, **kwargs) -> list[Union[CompoundStructurePair, DockingInputPair]]:
        ...

    def select(self, *args, **kwargs) -> list[Union[CompoundStructurePair, DockingInputPair]]:
        return self._select(*args, **kwargs)

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]:
        ...
