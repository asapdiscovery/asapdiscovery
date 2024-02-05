import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Optional, Union

import dask
import psutil
from asapdiscovery.data.enum import StringEnum
from asapdiscovery.data.execution_utils import (
    get_platform,
    guess_network_interface,
    hyperthreading_is_enabled,
)
from dask import config as cfg
from dask.utils import parse_timedelta
from dask_jobqueue import LSFCluster
from distributed import Client, LocalCluster
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BackendType(StringEnum):
    """
    Enum for backend types indicating how data is being passed into the dask function, either an in-memory object,
    or a JSON file of that object on disk
    """

    IN_MEMORY = "in-memory"
    DISK = "disk"


class DaskFailureMode(StringEnum):
    """
    Enum for Dask failure modes
    """

    RAISE = "raise"
    SKIP = "skip"


def set_dask_config():
    cfg.set({"distributed.scheduler.worker-ttl": None})
    cfg.set({"distributed.admin.tick.limit": "4h"})
    cfg.set({"distributed.scheduler.allowed-failures": 2})
    # do not tolerate failures, if work fails once job will be marked as permanently failed
    # this stops us cycling through jobs that fail losing all other work on the worker at that time
    cfg.set({"distributed.worker.memory.terminate": False})
    cfg.set({"distributed.worker.memory.pause": False})
    cfg.set({"distributed.worker.memory.target": 0.6})
    cfg.set({"distributed.worker.memory.spill": 0.7})
    cfg.set({"distributed.nanny.environ": {"MALLOC_TRIM_THRESHOLD_": 0}})


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


def backend_wrapper(
    inputs: Union[list[Any], list[Path]],
    func: Callable,
    backend=BackendType.IN_MEMORY,
    reconstruct_cls: Optional[Any] = None,
):
    if backend == BackendType.DISK:
        if not reconstruct_cls:
            raise ValueError("reconstruct_cls must be provided for disk backend")
        inputs = [reconstruct_cls.from_json_file(inp) for inp in inputs]
    elif backend == BackendType.IN_MEMORY:
        pass  # do nothing
    else:
        raise ValueError("invalid backend type")
    return func(inputs)


class DaskType(StringEnum):
    """
    Enum for Dask types
    """

    LOCAL = "local"
    LOCAL_GPU = "local-gpu"
    LILAC_GPU = "lilac-gpu"
    LILAC_CPU = "lilac-cpu"

    def is_lilac(self):
        return self in [DaskType.LILAC_CPU, DaskType.LILAC_GPU]


class GPU(StringEnum):
    """
    Enum for GPU types on lilac
    """

    GTX1080TI = "GTX1080TI"


class CPU(StringEnum):
    """
    Enum for CPU types
    """

    LT = "LT"


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


def dask_time_delta_diff(time_str_1: str, time_str_2: str) -> str:
    """
    Get the difference between two dask timedelta strings

    Parameters
    ----------
    time_str_1 : str
        A dask timedelta string, e.g. "1h30m"
    time_str_2 : str
        A dask timedelta string, e.g. "1h30m"

    Returns
    -------
    str
        A dask timedelta string that is the difference between time_str_1 and time_str_2 in seconds
    """
    seconds_1 = parse_timedelta(time_str_1)
    seconds_2 = parse_timedelta(time_str_2)
    seconds_diff = seconds_1 - seconds_2
    if seconds_diff < 0:
        raise ValueError(f"Time difference is negative: {seconds_diff}")
    return str(seconds_diff) + "s"


class DaskCluster(BaseModel):
    """
    Base config for a dask cluster

    Important is that cores == processes, otherwise we get some weird behaviour with multiple threads per process and
    OE heavy computation jobs.
    """

    class Config:
        allow_mutation = False
        extra = "forbid"

    name: str = Field("dask-worker", description="Name of the dask worker")
    cores: int = Field(8, description="Number of cores per job")
    processes: int = Field(8, description="Number of processes per job")
    memory: str = Field("48 GB", description="Amount of memory per job")
    death_timeout: int = Field(
        120, description="Timeout in seconds for a worker to be considered dead"
    )
    silence_logs: int = Field(logging.DEBUG, description="Log level for dask")


class LilacDaskCluster(DaskCluster):
    shebang: str = Field("#!/usr/bin/env bash", description="Shebang for the job")
    queue: str = Field("cpuqueue", description="LSF queue to submit jobs to")
    project: str = Field(None, description="LSF project to submit jobs to")
    walltime: str = Field("72h", description="Walltime for the job")
    use_stdin: bool = Field(True, description="Whether to use stdin for job submission")
    job_extra_directives: Optional[list[str]] = Field(
        None, description="Extra directives to pass to LSF"
    )
    job_script_prologue: list[str] = Field(
        ["ulimit -c 0"], description="Job prologue, default is to turn off core dumps"
    )
    dashboard_address: str = Field(":6412", description="port to activate dashboard on")
    lifetime_margin: str = Field(
        "10m",
        description="Margin to shut down workers before their walltime is up to ensure clean exit",
    )

    def to_cluster(self, exclude_interface: Optional[str] = "lo") -> LSFCluster:
        interface = guess_network_interface(exclude=[exclude_interface])
        _walltime = dask_timedelta_to_hh_mm(self.walltime)
        return LSFCluster(
            interface=interface,
            scheduler_options={
                "interface": interface,
                "dashboard_address": self.dashboard_address,
            },
            worker_extra_args=[
                "--lifetime",
                dask_time_delta_diff(self.walltime, self.lifetime_margin),
                "--lifetime-stagger",
                "10s",
            ],  # leave a buffer to cleanly exit
            walltime=_walltime,  # convert to LSF units manually
            **self.dict(exclude={"walltime", "dashboard_address", "lifetime_margin"}),
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
    def from_gpu(cls, gpu: GPU, **kwargs):
        return cls(
            gpu=gpu,
            gpu_group=_LILAC_GPU_GROUPS[gpu],
            extra=_LILAC_GPU_EXTRAS[gpu],
            **kwargs,
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
    def from_cpu(cls, cpu: CPU, **kwargs):
        return cls(cpu=cpu, cpu_group=_LILAC_CPU_GROUPS[cpu], extra=[], **kwargs)


class LilacGPUDaskCluster(LilacDaskCluster):
    queue: str = "gpuqueue"
    walltime = "72h"
    memory = "96 GB"
    cores = 1

    @classmethod
    def from_gpu(cls, gpu: GPU = GPU.GTX1080TI, silence_logs: int = logging.DEBUG):
        gpu_config = LilacGPUConfig.from_gpu(gpu)
        return cls(
            job_extra_directives=gpu_config.to_job_extra_directives(),
            silence_logs=silence_logs,
        )


class LilacCPUDaskCluster(LilacDaskCluster):
    # uses default

    @classmethod
    def from_cpu(cls, cpu: CPU = CPU.LT, silence_logs: int = logging.DEBUG):
        cpu_config = LilacCPUConfig.from_cpu(cpu)
        return cls(
            job_extra_directives=cpu_config.to_job_extra_directives(),
            silence_logs=silence_logs,
        )


def dask_cluster_from_type(
    dask_type: DaskType,
    gpu: GPU = GPU.GTX1080TI,
    cpu: CPU = CPU.LT,
    local_threads_per_worker: int = 1,
    loglevel: Union[str, int] = logging.DEBUG,
):
    """
    Get a dask client from a DaskType

    Parameters
    ----------
    dask_type : DaskType
        The type of dask client / cluster to get
    gpu : GPU, optional
        The GPU type to use if type is lilac-gpu, by default GPU.GTX1080TI
    cpu : CPU, optional
        The CPU type to use if type is lilac-cpu, by default CPU.LT

    Returns
    -------
    dask_jobqueue.Cluster
        A dask cluster
    """
    logger.info(f"Platform: {get_platform()}")
    cpu_count = psutil.cpu_count()
    logger.info(f"Logical CPU count: {cpu_count}")
    physical_cpu_count = psutil.cpu_count(logical=False)
    logger.info(f"Physical CPU count: {physical_cpu_count}")

    logger.info(f"Getting dask cluster of type {dask_type}")
    logger.info(f"Dask log level: {loglevel}")

    if dask_type == DaskType.LOCAL:
        n_workers = cpu_count // local_threads_per_worker
        logger.info(f"initial guess {n_workers} workers")
        if hyperthreading_is_enabled():
            logger.info("Hyperthreading is enabled")
            n_workers = n_workers // 2
            logger.info(f"Scaling to {n_workers} workers due to hyperthreading")
        else:
            logger.info("Hyperthreading is disabled")

        if n_workers < 1:
            n_workers = 1
            logger.info("Estimating 1 worker due to low CPU count")

        logger.info(f"Executing with {n_workers} workers")

        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=local_threads_per_worker,
            silence_logs=loglevel,  # used as silence_logs, worst kwarg name but it is what it is
        )
    elif dask_type == DaskType.LOCAL_GPU:
        try:
            from dask_cuda import LocalCUDACluster
        except ImportError:
            raise ImportError(
                "dask_cuda is not installed, please install with `pip install dask_cuda`"
            )
        cluster = LocalCUDACluster()
    elif dask_type == DaskType.LILAC_GPU:
        cluster = (
            LilacGPUDaskCluster()
            .from_gpu(gpu, silence_logs=loglevel)
            .to_cluster(exclude_interface="lo")
        )
    elif dask_type == DaskType.LILAC_CPU:
        cluster = (
            LilacCPUDaskCluster()
            .from_cpu(cpu, silence_logs=loglevel)
            .to_cluster(exclude_interface="lo")
        )
    else:
        raise ValueError(f"Unknown dask type {dask_type}")

    return cluster
