import abc
from typing import Literal, Union

import dask
from asapdiscovery.data.dask_utils import actualise_dask_delayed_iterable
from pydantic import BaseModel, Field
import pandas as pd


class VisualizerBase(abc.ABC, BaseModel):
    """
    Base class for visualizers.
    """

    @abc.abstractmethod
    def _visualize(self) -> list[Union[CompoundStructurePair, DockingInputPair]]:
        ...

    def visualize(
        self,
        docking_results: list[DockingInputPair],
        use_dask: bool = False,
        dask_client=None,
        **kwargs,
    ) -> pd.DataFrame:
        if use_dask:
            delayed_outputs = []
            for res in docking_results:
                out = dask.delayed(self._visualize)(docking_results=[res], **kwargs)
                delayed_outputs.append(out)
            outputs = actualise_dask_delayed_iterable(
                delayed_outputs, dask_client, errors="raise"
            )
            outputs = [item for sublist in outputs for item in sublist]  # flatten
        else:
            outputs = self._visualize(docking_results=docking_results, **kwargs)

        return outputs

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
    
    