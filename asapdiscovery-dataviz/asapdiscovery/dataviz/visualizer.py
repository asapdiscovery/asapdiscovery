import abc

import pandas as pd
from asapdiscovery.data.util.dask_utils import BackendType, FailureMode
from asapdiscovery.docking.docking import DockingResult
from pydantic.v1 import BaseModel


class VisualizerBase(abc.ABC, BaseModel):
    """
    Base class for visualizers.
    """

    @abc.abstractmethod
    def _visualize(self) -> pd.DataFrame: ...

    def visualize(
        self,
        inputs: list[DockingResult],
        use_dask: bool = False,
        dask_client=None,
        failure_mode=FailureMode.SKIP,
        backend=BackendType.IN_MEMORY,
        reconstruct_cls=None,
        **kwargs,
    ) -> pd.DataFrame:
        outputs = self._visualize(
            inputs=inputs,
            use_dask=use_dask,
            dask_client=dask_client,
            failure_mode=failure_mode,
            backend=backend,
            reconstruct_cls=reconstruct_cls,
            **kwargs,
        )

        return pd.DataFrame(outputs)

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]: ...
