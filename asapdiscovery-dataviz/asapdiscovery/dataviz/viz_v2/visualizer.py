import abc

import pandas as pd
from asapdiscovery.data.util.dask_utils import (
    BackendType,
    DaskFailureMode
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
        outputs = self._visualize(
            inputs=inputs,
            use_dask=use_dask,
            dask_client=dask_client,
            dask_failure_mode=dask_failure_mode,
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
