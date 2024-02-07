import abc

import dask
import pandas as pd
from asapdiscovery.data.util.dask_utils import (
    BackendType,
    DaskFailureMode,
    actualise_dask_delayed_iterable,
    backend_wrapper,
)
from asapdiscovery.docking.docking import DockingResult
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
        inputs: list[DockingResult],
        use_dask: bool = False,
        dask_client=None,
        dask_failure_mode=DaskFailureMode.SKIP,
        backend=BackendType.IN_MEMORY,
        reconstruct_cls=None,
    ) -> pd.DataFrame:
        if use_dask:
            delayed_outputs = []
            for inp in inputs:
                out = dask.delayed(backend_wrapper)(
                    inputs=[inp],
                    func=self._visualize,
                    backend=backend,
                    reconstruct_cls=reconstruct_cls,
                )
                delayed_outputs.append(out[0])  # flatten
            outputs = actualise_dask_delayed_iterable(
                delayed_outputs, dask_client, errors=dask_failure_mode
            )
        else:
            outputs = backend_wrapper(
                inputs=inputs,
                func=self._visualize,
                backend=backend,
                reconstruct_cls=reconstruct_cls,
            )

        return pd.DataFrame(outputs)

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
