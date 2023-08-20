from enum import Enum
from typing import Optional

from dask import config as cfg
from dask.utils import parse_timedelta
from dask_jobqueue import LSFCluster
from pydantic import BaseModel, Field, validator

from .execution_utils import guess_network_interface

cfg.set({"distributed.scheduler.worker-ttl": None})
cfg.set({"distributed.admin.tick.limit": "2h"})


class GPU(str, Enum):
    """
    Enum for GPU types
    """

    GTX1080TI = "GTX1080TI"

    @classmethod
    def get_values(cls):
        return [gpu.value for gpu in cls]


_LILAC_GPU_GROUPS = {
    GPU.GTX1080TI: "lt-gpu",
}

_LILAC_GPU_EXTRAS = {
    GPU.GTX1080TI: [
        '-R "select[hname!=lt16]"'
    ],  # random node that has something wrong with its GPU driver versioning
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
    cores: int = Field(1, description="Number of cores per job")
    memory: str = Field("20 GB", description="Amount of memory per job")
    death_timeout: int = Field(
        120, description="Timeout in seconds for a worker to be considered dead"
    )


class LilacDaskCluster(DaskCluster):
    shebang: str = Field("#!/usr/bin/env bash", description="Shebang for the job")
    queue: str = Field("cpuqueue", description="LSF queue to submit jobs to")
    project: str = Field(None, description="LSF project to submit jobs to")
    walltime: str = Field("1h", description="Walltime for the job")
    use_stdin: bool = Field(True, description="Whether to use stdin for job submission")
    job_extra_directives: Optional[list[str]] = Field(
        None, description="Extra directives to pass to LSF"
    )

    @validator("walltime")
    @classmethod
    def _convert_dask_time(cls, v):
        return dask_timedelta_to_hh_mm(v)

    def to_cluster(self, exclude_interface: Optional[str] = "lo") -> LSFCluster:
        interface = guess_network_interface(exclude=[exclude_interface])
        return LSFCluster(
            interface=interface,
            scheduler_options={"interface": interface},
            worker_extra_args=[
                "--lifetime",
                f"{self.walltime}",
                "--lifetime-stagger",
                "2m",
            ],  # leave a slight buffer
            **self.dict(),
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


class LilacGPUDaskCluster(LilacDaskCluster):
    queue: str = "gpuqueue"
    walltime = "24h"
    memory = "48 GB"

    @classmethod
    def from_gpu(cls, gpu: GPU = GPU.GTX1080TI):
        gpu_config = LilacGPUConfig.from_gpu(gpu)
        return cls(job_extra_directives=gpu_config.to_job_extra_directives())
