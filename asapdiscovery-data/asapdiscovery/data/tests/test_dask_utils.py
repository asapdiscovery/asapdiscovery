from asapdiscovery.data.dask_utils import LilacDaskCluster, LilacGPUDaskCluster


def test_LilacDaskCluster():
    opts = LilacDaskCluster()
    cluster = opts.to_cluster()
    js = cluster.job_script()
    assert "dask-worker" in js


def test_LilacGPUDDaskCluster():
    opts = LilacGPUDaskCluster()
    cluster = opts.to_cluster()
    js = cluster.job_script()
    assert "dask-worker" in js
    assert "gpuqueue" in js


def test_LilacGPUDaskCluster_from_gpu():
    opts = LilacGPUDaskCluster.from_gpu("GTX1080TI")
    cluster = opts.to_cluster()
    js = cluster.job_script()
    assert "dask-worker" in js
    assert "gpuqueue" in js
    assert "lt-gpu" in js
    assert '-R "select[hname!=lt16]"' in js
