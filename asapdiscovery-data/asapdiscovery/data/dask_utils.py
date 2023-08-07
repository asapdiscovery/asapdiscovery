from dask.distributed import Client, LocalCluster
from dask_jobqueue import LSFCluster
from pydantic import BaseModel, Field
from typing import Optional
from .execution_utils import guess_network_interface
from enum import Enum
from dask import config as cfg

cfg.set({"distributed.scheduler.worker-ttl": None})
cfg.set({"distributed.admin.tick.limit": "2h"})


class GPU(str, Enum):
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
    ],  # random node that has something wrong with its GPU
}


def _walltime_to_h(walltime: str) -> int:
    """
    Convert a walltime string to hours, dropping minutes and seconds.
    """
    return int(walltime.split(":")[0])

class DaskCluster(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"

    name: str = Field("dask-worker", description="Name of the dask worker")
    cores: int = Field(1, description="Number of cores per job")
    memory: str = Field("20 GB", description="Amount of memory per job")
    death_timeout: int = Field(
        120, description="Timeout for a worker to be considered dead"
    )


class LilacDaskCluster(DaskCluster):
    shebang: str = Field("#!/usr/bin/env bash", description="Shebang for the job")
    queue: str = Field("cpuqueue", description="LSF queue to submit jobs to")
    project: str = Field(None, description="LSF project to submit jobs to")
    walltime: str = Field("1:00", description="Walltime for the job")
    use_stdin: bool = Field(True, description="Whether to use stdin for job submission")
    job_extra_directives: Optional[list[str]] = Field(
        None, description="Extra directives to pass to LSF"
    )

    def to_cluster(self, exclude_interface: Optional[str] = "lo"):
        interface = guess_network_interface(exclude=[exclude_interface])
        return LSFCluster(
            interface=interface,
            scheduler_options={"interface": interface},
            worker_extra_args=["--lifetime", f"{_walltime_to_h(self.walltime) - 1}", "--lifetime-stagger", "2m"], # leave a slight buffer
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
    walltime = "24:00"
    memory = "48 GB"

    @classmethod
    def from_gpu(cls, gpu: GPU = GPU.GTX1080TI):
        gpu_config = LilacGPUConfig.from_gpu(gpu)
        return cls(job_extra_directives=gpu_config.to_job_extra_directives())
