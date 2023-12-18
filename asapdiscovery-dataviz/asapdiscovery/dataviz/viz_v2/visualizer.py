import abc
from pathlib import Path
from typing import Union

import dask
import pandas as pd
from asapdiscovery.data.dask_utils import (
    BackendType,
    actualise_dask_delayed_iterable,
    dask_backend_wrapper,
)
from asapdiscovery.docking.docking_v2 import DockingResult
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
        reconstruct_cls=None,
        **kwargs,
    ) -> pd.DataFrame:
        if use_dask:
            delayed_outputs = []
            for res in docking_results:
                out = dask.delayed(dask_backend_wrapper)(
                    inputs=[res],
                    func=self._visualize,
                    backend=backend,
                    reconstruct_cls=reconstruct_cls,
                )
                delayed_outputs.append(out)
            outputs = actualise_dask_delayed_iterable(
                delayed_outputs, dask_client, errors="skip"
            )
            outputs = [item for sublist in outputs for item in sublist]  # flatten
        else:
            outputs = self._visualize(docking_results=docking_results, *args, **kwargs)

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
