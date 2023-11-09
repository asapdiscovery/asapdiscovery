import abc
from typing import Literal, Union

import dask
from asapdiscovery.data.dask_utils import actualise_dask_delayed_iterable
from asapdiscovery.data.schema_v2.complex import Complex, PreppedComplex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.pairs import CompoundStructurePair
from asapdiscovery.docking.docking_v2 import DockingInputPair
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
    def _select(self) -> list[Union[CompoundStructurePair, DockingInputPair]]:
        ...

    def select(
        self,
        ligands: list[Ligand],
        complexes: list[Union[Complex, PreppedComplex]],
        use_dask: bool = False,
        dask_client=None,
        **kwargs,
    ) -> list[Union[CompoundStructurePair, DockingInputPair]]:
        if use_dask:
            delayed_outputs = []
            for lig in ligands:
                out = dask.delayed(self._select)(
                    ligands=[lig], complexes=complexes, **kwargs
                )  # be careful here, need ALL complexes to perform a full search, ie no parallelism over complexes is possible.
                delayed_outputs.append(out)
            outputs = actualise_dask_delayed_iterable(
                delayed_outputs, dask_client, errors="raise"
            )
            outputs = [
                item for sublist in outputs for item in sublist
            ]  # flatten post hoc
        else:
            outputs = self._select(ligands=ligands, complexes=complexes, **kwargs)

        return outputs

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
