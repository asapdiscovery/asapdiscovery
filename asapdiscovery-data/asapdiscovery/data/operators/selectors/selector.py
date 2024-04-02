import abc
from typing import Literal, Union

import dask
from asapdiscovery.data.schema.complex import Complex, PreppedComplex
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.schema.pairs import CompoundStructurePair
from asapdiscovery.data.util.dask_utils import (
    DaskFailureMode,
    actualise_dask_delayed_iterable,
)
from asapdiscovery.docking.docking import DockingInputPair  # TODO: move to backend
from pydantic import BaseModel


class SelectorBase(abc.ABC, BaseModel):
    """
    Base class for selectors.
    """

    # records what kind of selector class was used, overridden in subclasses
    @abc.abstractmethod
    def selector_type(self) -> str: ...

    @abc.abstractmethod
    def _select(self) -> list[Union[CompoundStructurePair, DockingInputPair]]: ...

    def select(
        self,
        ligands: list[Ligand],
        complexes: list[Union[Complex, PreppedComplex]],
        use_dask: bool = False,
        dask_client=None,
        dask_failure_mode=DaskFailureMode.SKIP,
        **kwargs,
    ) -> list[Union[CompoundStructurePair, DockingInputPair]]:
        if use_dask:
            delayed_outputs = []
            for lig in ligands:
                out = dask.delayed(self._select)(
                    ligands=[lig], complexes=complexes, **kwargs
                )  # be careful here, need ALL complexes to perform a full search, ie no parallelism over complexes is possible with current setup
                # see # 560
                delayed_outputs.append(out)
            outputs = actualise_dask_delayed_iterable(
                delayed_outputs, dask_client, errors=dask_failure_mode
            )
            outputs = [
                item for sublist in outputs for item in sublist
            ]  # flatten post hoc
        else:
            outputs = self._select(ligands=ligands, complexes=complexes, **kwargs)

        return outputs

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]: ...

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
