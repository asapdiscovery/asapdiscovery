import abc
from typing import Union
import dask
import pandas as pd
from pathlib import Path
from asapdiscovery.data.dask_utils import actualise_dask_delayed_iterable, BackendType
from asapdiscovery.docking.docking_v2 import DockingResult
from asapdiscovery.docking.openeye import POSITDockingResults
from pydantic import BaseModel


class VisualizerBase(abc.ABC, BaseModel):
    """
    Base class for visualizers.
    """

    @abc.abstractmethod
    def _visualize(self) -> pd.DataFrame:
        ...

    def visualize(
        self,
        docking_results: list[DockingResult],
        *args,
        use_dask: bool = False,
        dask_client=None,
        backend=BackendType.IN_MEMORY,
        **kwargs,
    ) -> pd.DataFrame:
        if use_dask:
            delayed_outputs = []
            for res in docking_results:
                out = dask.delayed(self._dask_wrapper)(
                    docking_results=[res], backend=backend, **kwargs
                )
                delayed_outputs.append(out)
            outputs = actualise_dask_delayed_iterable(
                delayed_outputs, dask_client, errors="raise"
            )
            outputs = [item for sublist in outputs for item in sublist]  # flatten
        else:
            outputs = self._visualize(docking_results=docking_results, *args, **kwargs)

        return pd.DataFrame(outputs)

    # TODO: this is a bit hacky, but it works
    # workaround to create data on workers rather than passing it
    def _dask_wrapper(
        self,
        docking_results: Union[list[DockingResult], list[Path]],
        backend=BackendType.IN_MEMORY,
    ):
        if backend == BackendType.DISK:
            docking_results = [
                POSITDockingResults.from_json_file(inp) for inp in docking_results
            ]
        elif backend == BackendType.IN_MEMORY:
            pass  # do nothing
        else:
            raise Exception("invalid backend type")

        return self._visualize(docking_results=docking_results)

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]:
        ...

    @staticmethod
    def write_data(data, path):
        """
        Write data to a file.

        Parameters
        ----------
        data : str
            data to write.
        path : Path
            Path to write HTML to.
        """
        with open(path, "w") as f:
            f.write(data)
