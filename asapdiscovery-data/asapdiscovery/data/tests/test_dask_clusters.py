import time

import pytest
from dask.distributed import Client

from asapdiscovery.data.util.dask_utils import DaskType, make_dask_client_meta


@pytest.mark.parametrize("loglevel", ["DEBUG", 10])
@pytest.mark.parametrize("type", [DaskType("local")])
def test_make_dask_cluster_meta(type, loglevel):
    meta = make_dask_client_meta(type, loglevel=loglevel)
    assert isinstance(meta, Client)
    time.sleep(2)
    meta.close()
