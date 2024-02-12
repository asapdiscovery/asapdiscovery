import pytest
from asapdiscovery.data.dask_utils import (
    DaskCluster,
    DaskType,
    LilacCPUConfig,
    LilacCPUDaskCluster,
    LilacDaskCluster,
    LilacGPUConfig,
    LilacGPUDaskCluster,
    make_dask_client_meta,
)


def test_dask_cluster():
    cluster = DaskCluster()
    assert cluster.name == "dask-worker"
    assert cluster.cores == 8
    assert cluster.memory == "48 GB"


def test_lilac_dask_cluster():
    cluster = LilacDaskCluster()
    assert cluster.use_stdin == True
    assert cluster.queue == "cpuqueue"


def test_lilac_gpu_config():
    config = LilacGPUConfig.from_gpu("GTX1080TI")
    assert config.gpu == "GTX1080TI"
    assert config.gpu_group == "lt-gpu"


def test_lilac_cpu_config():
    config = LilacCPUConfig.from_cpu("LT")
    assert config.cpu_group == "lt-gpu"  # CPU group is the same as GPU group


@pytest.mark.parametrize("loglevel", ["DEBUG", 10])
def test_lilac_gpu_cluster(loglevel):
    cluster = LilacGPUDaskCluster.from_gpu("GTX1080TI", loglevel=loglevel)
    assert cluster is not None


@pytest.mark.parametrize("loglevel", ["DEBUG", 10])
def test_lilac_cpu_cluster(loglevel):
    cluster = LilacCPUDaskCluster.from_cpu("LT", loglevel=loglevel)
    assert cluster is not None


@pytest.mark.parametrize("loglevel", ["DEBUG", 10])
@pytest.mark.parametrize(
    "type", [DaskType("local"), DaskType("lilac-cpu"), DaskType("lilac-gpu")]
)
def test_make_dask_cluster_meta(type, loglevel):
    meta = make_dask_client_meta(type, loglevel=loglevel)
    assert meta is not None
