from collections.abc import Iterable
from enum import Enum
from typing import Optional

import dask
from dask import config as cfg
from dask.utils import parse_timedelta
from dask_jobqueue import LSFCluster
from distributed import Client, LocalCluster
from pydantic import BaseModel, Field

from .execution_utils import guess_network_interface


def set_dask_config():
    cfg.set({"distributed.scheduler.worker-ttl": None})
    cfg.set({"distributed.admin.tick.limit": "2h"})
    cfg.set({"distributed.scheduler.active-memory-manager.measure": "managed"})
    cfg.set({"distributed.worker.memory.rebalance.measure": "managed"})
    cfg.set({"distributed.worker.memory.spill": False})
    cfg.set({"distributed.worker.memory.pause": True})
    cfg.set({"distributed.worker.memory.terminate": False})


def actualise_dask_delayed_iterable(
    delayed_iterable: Iterable,
    dask_client: Optional[Client] = None,
    errors: str = "raise",
):
    """
    Run a list of dask delayed functions or collections, and return the results
    If a dask client is provided is run as a future, otherwise as a compute
    Parameters
    ----------
    delayed_iterable : Iterable
        List of dask delayed functions
    dask_client Client, optional
        Dask client to use, by default None
    Returns
    -------
    iterable: Iterable
        Iterable of computed results from the dask delayed functions
    """
    if dask_client is None:
        return dask.compute(*delayed_iterable)
    else:
        futures = dask_client.compute(delayed_iterable)
    return dask_client.gather(futures, errors=errors)


class DaskType(str, Enum):
    """
    Enum for Dask types
    """

    LOCAL = "local"
    LILAC_GPU = "lilac-gpu"
    LILAC_CPU = "lilac-cpu"

    @classmethod
    def get_values(cls):
        return [dask_type.value for dask_type in cls]

    def is_lilac(self):
        return self in [DaskType.LILAC_CPU, DaskType.LILAC_GPU]


class GPU(str, Enum):
    """
    Enum for GPU types
    """

    GTX1080TI = "GTX1080TI"

    @classmethod
    def get_values(cls):
        return [gpu.value for gpu in cls]


class CPU(str, Enum):
    """
    Enum for CPU types
    """

    LT = "LT"

    @classmethod
    def get_values(cls):
        return [cpu.value for cpu in cls]


_LILAC_GPU_GROUPS = {
    GPU.GTX1080TI: "lt-gpu",
}

_LILAC_GPU_EXTRAS = {
    GPU.GTX1080TI: [
        '-R "select[hname!=lt16]"'
    ],  # random node that has something wrong with its GPU driver versioning
}

# Lilac does not have a specific CPU group, but we can use the lt-gpu group
# for CPU jobs in combination with `cpuqueue` (per MSK HPC) as lt has the most nodes.
_LILAC_CPU_GROUPS = {
    CPU.LT: "lt-gpu",
}


def dask_timedelta_to_hh_mm(time_str: str) -> str:
    """
    Convert a dask timedelta string to a hh:mm string

    Parameters
    ----------
    time_str : str
        A dask timedelta string, e.g. "1h30m"

    Returns
    -------
    str
        A string in the format hh:mm, e.g. "01:30"
    """
    seconds = parse_timedelta(time_str)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"


class DaskCluster(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"

    name: str = Field("dask-worker", description="Name of the dask worker")
    cores: int = Field(8, description="Number of cores per job")
    memory: str = Field("24 GB", description="Amount of memory per job")
    death_timeout: int = Field(
        120, description="Timeout in seconds for a worker to be considered dead"
    )


class LilacDaskCluster(DaskCluster):
    shebang: str = Field("#!/usr/bin/env bash", description="Shebang for the job")
    queue: str = Field("cpuqueue", description="LSF queue to submit jobs to")
    project: str = Field(None, description="LSF project to submit jobs to")
    walltime: str = Field("24h", description="Walltime for the job")
    use_stdin: bool = Field(True, description="Whether to use stdin for job submission")
    job_extra_directives: Optional[list[str]] = Field(
        None, description="Extra directives to pass to LSF"
    )
    job_script_prologue: list[str] = Field(
        ["ulimit -c 0"], description="Job prologue, default is to turn off core dumps"
    )

    def to_cluster(self, exclude_interface: Optional[str] = "lo") -> LSFCluster:
        interface = guess_network_interface(exclude=[exclude_interface])
        _walltime = dask_timedelta_to_hh_mm(self.walltime)
        return LSFCluster(
            interface=interface,
            scheduler_options={"interface": interface},
            worker_extra_args=[
                "--lifetime",
                f"{self.walltime}",
                "--lifetime-stagger",
                "2m",
            ],  # leave a slight buffer
            walltime=_walltime,  # convert to LSF units manually
            **self.dict(exclude={"walltime"}),
        )


class LilacGPUConfig(BaseModel):
    gpu: GPU = Field(..., description="GPU type")
    gpu_group: str = Field(..., description="GPU group")
    extra: Optional[list[str]] = Field(
        None, description="Extra directives to pass to LSF"
    )

    def to_job_extra_directives(self):
        return [
            "-gpu num=1:j_exclusive=yes:mode=shared",
            f"-m {self.gpu_group}",
            *self.extra,
        ]

    @classmethod
    def from_gpu(cls, gpu: GPU):
        return cls(
            gpu=gpu, gpu_group=_LILAC_GPU_GROUPS[gpu], extra=_LILAC_GPU_EXTRAS[gpu]
        )


class LilacCPUConfig(BaseModel):
    cpu: CPU = Field(..., description="CPU type")
    cpu_group: str = Field(..., description="CPU group")
    extra: Optional[list[str]] = Field(
        None, description="Extra directives to pass to LSF"
    )

    def to_job_extra_directives(self):
        return [
            f"-m {self.cpu_group}",
            *self.extra,
        ]

    @classmethod
    def from_cpu(cls, cpu: CPU):
        return cls(cpu=cpu, cpu_group=_LILAC_CPU_GROUPS[cpu], extra=[])


class LilacGPUDaskCluster(LilacDaskCluster):
    queue: str = "gpuqueue"
    walltime = "24h"
    memory = "48 GB"
    cores = 1

    @classmethod
    def from_gpu(cls, gpu: GPU = GPU.GTX1080TI):
        gpu_config = LilacGPUConfig.from_gpu(gpu)
        return cls(job_extra_directives=gpu_config.to_job_extra_directives())


class LilacCPUDaskCluster(LilacDaskCluster):
    # uses default

    @classmethod
    def from_cpu(cls, cpu: CPU = CPU.LT):
        cpu_config = LilacCPUConfig.from_cpu(cpu)
        return cls(job_extra_directives=cpu_config.to_job_extra_directives())


def dask_cluster_from_type(
    dask_type: DaskType, gpu: GPU = GPU.GTX1080TI, cpu: CPU = CPU.LT
):
    """
    Get a dask client from a DaskType

    Parameters
    ----------
    dask_type : DaskType
        The type of dask client / cluster to get
    gpu : GPU, optional
        The GPU type to use if type is GPU, by default GPU.GTX1080TI
    cpu : CPU, optional
        The CPU type to use if type is CPU, by default CPU.LT

    Returns
    -------
    dask_jobqueue.Cluster
        A dask cluster
    """
    if dask_type == DaskType.LOCAL:
        cluster = LocalCluster()
    elif dask_type == DaskType.LILAC_GPU:
        cluster = LilacGPUDaskCluster().from_gpu(gpu).to_cluster(exclude_interface="lo")
    elif dask_type == DaskType.LILAC_CPU:
        cluster = LilacCPUDaskCluster().from_cpu(cpu).to_cluster(exclude_interface="lo")
    else:
        raise ValueError(f"Unknown dask type {dask_type}")

    return cluster
