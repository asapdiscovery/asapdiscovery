import abc
from typing import Literal, Union

from asapdiscovery.data.schema_v2.pairs import CompoundStructurePair, DockingInputPair
from asapdiscovery.data.schema_v2.complex import Complex, PreppedComplex


from pydantic import BaseModel, Field


class SelectorBase(abc.ABC, BaseModel):
    """
    Base class for selectors.
    """

    # records what kind of selector class was used, overridden in subclasses
    selector_type: Literal["SelectorBase"] = Field(
        "SelectorBase", description="The type of selector to use"
    )

    @abc.abstractmethod
    def _select(
        self, *args, **kwargs
    ) -> list[Union[CompoundStructurePair, DockingInputPair]]:
        ...

    def select(
        self, *args, **kwargs
    ) -> list[Union[CompoundStructurePair, DockingInputPair]]:
        return self._select(*args, **kwargs)

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]:
        ...

    @staticmethod
    def _pair_type_from_complex(
        complex: Union[Complex, PreppedComplex]
    ) -> Literal["CompoundStructurePair", "DockingInputPair"]:
        """
        Returns the pair type that matches a given Complex type.
        Complex -> CompoundStructurePair
        PreppedComplex -> DockingInputPair

        Parameters
        ----------
        complex : Union[Complex, PreppedComplex]
            Complex to get pair type for

        Returns
        -------
        """
        if isinstance(complex, Complex):
            return CompoundStructurePair
        elif isinstance(complex, PreppedComplex):
            return DockingInputPair
        else:
            raise ValueError(f"Unknown complex type: {type(complex)}")
