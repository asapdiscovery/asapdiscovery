import functools
import logging
from collections.abc import Iterable
from typing import Optional, Union

import dask
import numpy as np
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
    errors: str = DaskFailureMode.RAISE.value,
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


def backend_wrapper(kwargname):
    """
    Decorator to handle dask backend for passing data into a function from disk or in-memory
    kwargname is the name of the keyword argument that is being passed in from disk or in-memory

    The decorator will take the following kwargs from a call site and pops them from kwargs.


    Parameters
    ----------
    backend : BackendType
        The backend type to use, either in-memory or disk
    reconstruct_cls : Callable
        The class to use to reconstruct the object from disk
    """

    def backend_wrapper_inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            backend = kwargs.pop("backend", None)
            reconstruct_cls = kwargs.pop("reconstruct_cls", None)

            if backend == BackendType.DISK:
                # grab optional disk kwargs
                to_be_reconstructed = kwargs.pop(kwargname, None)
                if to_be_reconstructed is None:
                    raise ValueError(f"Missing keyword argument {kwargname}")

                if reconstruct_cls is None:
                    raise ValueError("Missing keyword argument reconstruct_cls")

                # reconstruct the object from disk
                reconstructed = [
                    reconstruct_cls.from_json_file(f) for f in to_be_reconstructed
                ]

                # add the reconstructed object to the kwargs
                kwargs[kwargname] = reconstructed

            elif backend == BackendType.IN_MEMORY:
                pass
            else:
                raise ValueError(f"Unknown backend type {backend}")
            return func(*args, **kwargs)

        return wrapper

    return backend_wrapper_inner


def dask_vmap(kwargsnames):
    """
    Decorator to handle either returning a whole vector if not using dask, or using dask to parallelise over a vector
    if dask is being used

    Designed to be used structure of the form

    @dask_vmap(["kwargs1", "kwargs2"])
    def my_function(kwargs1, kwargs2, use_dask=False, dask_client=None, dask_failure_mode=DaskFailureMode.RAISE.value):
        return _my_function(kwargs1, kwargs2)

    If use_dask is `True`, then `_my_function` will be parallelised over kwargs1 and kwargs2 (zipped, must be same length) using dask, passing in iterable
    intputs of length 1. If use_dask is `False` it will call `_my_function` directly.

    Parameters
    ----------
    kwargsnames : list[str]
        List of keyword argument names to parallelise over
    use_dask : bool, optional
        Whether to use dask, by default False
    dask_client : Client, optional
        Dask client to use, by default None
    dask_failure_mode : str, optional
        Dask failure mode, by default DaskFailureMode.RAISE.value
    """

    def dask_vmap_inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # grab optional dask kwargs
            use_dask = kwargs.pop("use_dask", None)
            dask_client = kwargs.pop("dask_client", None)
            dask_failure_mode = kwargs.pop(
                "dask_failure_mode", DaskFailureMode.SKIP.value
            )

            if use_dask:
                # grab iterable_kwargs
                iterable_kwargs = {name: kwargs.pop(name) for name in kwargsnames}
                # check they are all the same length
                # Check if all iterable keyword arguments are of the same length
                lengths = {name: len(value) for name, value in iterable_kwargs.items()}
                if len(set(lengths.values())) != 1:
                    raise ValueError(
                        "Iterable keyword arguments must be of the same length."
                    )

                computations = []
                for values in zip(*iterable_kwargs.values()):
                    local_kwargs = kwargs.copy()
                    for name, value in zip(iterable_kwargs.keys(), values):
                        local_kwargs[name] = [value]
                    computations.append(dask.delayed(func)(*args, **local_kwargs))
                return np.ravel(
                    actualise_dask_delayed_iterable(
                        computations, dask_client=dask_client, errors=dask_failure_mode
                    )
                ).tolist()
            else:
                return func(*args, **kwargs)

        return wrapper

    return dask_vmap_inner


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

    # lilcac lt-gpu queue used in CPU mode
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
    silence_logs: Union[int, str] = Field(
        logging.INFO, description="Log level for dask"
    )


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
    def from_gpu(
        cls,
        gpu: GPU = GPU.GTX1080TI,
        loglevel: Union[str, int] = logging.INFO,
        walltime: str = "72h",
    ):
        gpu_config = LilacGPUConfig.from_gpu(gpu)
        return cls(
            job_extra_directives=gpu_config.to_job_extra_directives(),
            silence_logs=loglevel,
            walltime=walltime,
        )


class LilacCPUDaskCluster(LilacDaskCluster):
    # uses default

    @classmethod
    def from_cpu(
        cls,
        cpu: CPU = CPU.LT,
        loglevel: Union[int, str] = logging.INFO,
        walltime: str = "72h",
    ):
        cpu_config = LilacCPUConfig.from_cpu(cpu)
        return cls(
            job_extra_directives=cpu_config.to_job_extra_directives(),
            silence_logs=loglevel,
            walltime=walltime,
        )


def dask_cluster_from_type(
    dask_type: DaskType,
    gpu: GPU = GPU.GTX1080TI,
    cpu: CPU = CPU.LT,
    local_threads_per_worker: int = 1,
    loglevel: Union[int, str] = logging.INFO,
    walltime: str = "72h",
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
            .from_gpu(gpu, loglevel=loglevel, walltime=walltime)
            .to_cluster(exclude_interface="lo")
        )
    elif dask_type == DaskType.LILAC_CPU:
        cluster = (
            LilacCPUDaskCluster()
            .from_cpu(cpu, loglevel=loglevel, walltime=walltime)
            .to_cluster(exclude_interface="lo")
        )
    else:
        raise ValueError(f"Unknown dask type {dask_type}")

    return cluster


def make_dask_client_meta(
    dask_type: DaskType,
    loglevel: Union[int, str] = logging.INFO,
    walltime: str = "72h",
    adaptive_min_workers: int = 10,
    adaptive_max_workers: int = 200,
    adaptive_wait_count: int = 10,
    adaptive_interval: str = "1m",
):
    logger.info(f"Using dask for parallelism of type: {dask_type}")
    if isinstance(loglevel, int):
        loglevel = logging.getLevelName(loglevel)
    set_dask_config()
    dask_cluster = dask_cluster_from_type(
        dask_type, loglevel=loglevel, walltime=walltime
    )
    if dask_type.is_lilac():
        logger.info("Lilac HPC config selected, setting adaptive scaling")
        dask_cluster.adapt(
            minimum=adaptive_min_workers,
            maximum=adaptive_max_workers,
            wait_count=adaptive_wait_count,
            interval=adaptive_interval,
        )
        logger.info(
            f"Starting with minimum worker count: {adaptive_min_workers} workers"
        )
        dask_cluster.scale(adaptive_min_workers)

    dask_client = Client(dask_cluster)
    dask_client.forward_logging(level=loglevel)
    logger.info(f"Using dask client: {dask_client}")
    logger.info(f"Using dask cluster: {dask_cluster}")
    logger.info(f"Dask client dashboard: {dask_client.dashboard_link}")

    return dask_client
