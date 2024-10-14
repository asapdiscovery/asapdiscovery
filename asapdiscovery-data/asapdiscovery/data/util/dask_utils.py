import functools
import itertools
import logging
from collections.abc import Iterable
from typing import Optional, Union

import dask
import psutil
from asapdiscovery.data.util.execution_utils import (
    get_platform,
    hyperthreading_is_enabled,
)
from asapdiscovery.data.util.stringenum import StringEnum
from dask import config as cfg
from dask.utils import parse_timedelta
from distributed import Client, LocalCluster

logger = logging.getLogger(__name__)


class BackendType(StringEnum):
    """
    Enum for backend types indicating how data is being passed into the dask function, either an in-memory object,
    or a JSON file of that object on disk
    """

    IN_MEMORY = "in-memory"
    DISK = "disk"


class FailureMode(StringEnum):
    """
    Enum for failure modes
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
    errors: str = FailureMode.RAISE.value,
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
    errors : str
        Dask failure mode, one of "raise" or "skip", by default "raise"
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


def backend_wrapper(kwargname, pop_kwargs=True):
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
    pop_kwargs : bool, optional
        Whether to pop the kwargs from the kwargs dict, by default True
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

                if not pop_kwargs:
                    # restore the kwargs
                    kwargs["backend"] = backend
                    kwargs["reconstruct_cls"] = reconstruct_cls

            elif backend == BackendType.IN_MEMORY:
                pass
            else:
                raise ValueError(f"Unknown backend type {backend}")
            return func(*args, **kwargs)

        return wrapper

    return backend_wrapper_inner


def dask_vmap(kwargsnames, remove_falsy=True, has_failure_mode=False):
    """
    Decorator to handle either returning a whole vector if not using dask, or using dask to parallelise over a vector
    if dask is being used

    Designed to be used structure of the form

    @dask_vmap(["kwargs1", "kwargs2"])
    def my_function(kwargs1, kwargs2, use_dask=False, dask_client=None, failure_mode=FailureMode.RAISE.value):
        return _my_function(kwargs1, kwargs2)

    If use_dask is `True`, then `_my_function` will be parallelised over kwargs1 and kwargs2 (zipped, must be same length) using dask, passing in iterable
    intputs of length 1. If use_dask is `False` it will call `_my_function` directly.

    Parameters
    ----------
    kwargsnames : list[str]
        List of keyword argument names to parallelise over
    remove_falsy : bool, optional
        Whether to remove falsy (bool casted) results from the output, by default True
    use_dask : bool, optional
        Whether to use dask, by default False
    dask_client : Client, optional
        Dask client to use, by default None
    failure_mode : str, optional
        Dask failure mode, by default FailureMode.RAISE.value
    """

    def dask_vmap_inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # grab optional dask kwargs
            use_dask = kwargs.pop("use_dask", None)
            dask_client = kwargs.pop("dask_client", None)
            failure_mode = kwargs.pop("failure_mode", FailureMode.SKIP.value)

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
                    if has_failure_mode:
                        local_kwargs["failure_mode"] = failure_mode
                    for name, value in zip(iterable_kwargs.keys(), values):
                        local_kwargs[name] = [value]
                    computations.append(dask.delayed(func)(*args, **local_kwargs))

                results = actualise_dask_delayed_iterable(
                    computations, dask_client=dask_client, errors=failure_mode
                )

                if remove_falsy:
                    results = [r for r in results if r]

                # flatten possibly ragged list of lists
                results = list(itertools.chain(*results))
                return results

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


def dask_cluster_from_type(
    dask_type: DaskType,
    threads_per_worker: int = 1,
    loglevel: Union[int, str] = logging.INFO,
    n_workers: int = None,
):
    """
    Get a dask client from a DaskType

    Parameters
    ----------
    dask_type : DaskType
        The type of dask client / cluster to get
    local_threads_per_worker : int, optional
        The number of threads per worker for a local cluster, by default 1
    loglevel : Union[int, str], optional
        The log level to use, by default logging.INFO
    force_n_workers : int, optional
        The number of workers to use, by default None

    Returns
    -------
    dask.Cluster
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
        if n_workers is not None:
            pass
        else:
            n_workers = cpu_count // threads_per_worker
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
            threads_per_worker=threads_per_worker,
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
    else:
        raise ValueError(f"Unknown dask type {dask_type}")

    return cluster


def make_dask_client_meta(
    dask_type: DaskType,
    loglevel: Union[int, str] = logging.INFO,
    n_workers: int = None,
    threads_per_worker: int = 1,
):
    logger.info(f"Using dask for parallelism of type: {dask_type}")
    if isinstance(loglevel, int):
        loglevel = logging.getLevelName(loglevel)
    set_dask_config()
    dask_cluster = dask_cluster_from_type(
        dask_type,
        loglevel=loglevel,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
    )
    dask_client = Client(dask_cluster)
    dask_client.forward_logging(level=loglevel)
    logger.info(f"Using dask client: {dask_client}")
    logger.info(f"Using dask cluster: {dask_cluster}")
    logger.info(f"Dask client dashboard: {dask_client.dashboard_link}")

    return dask_client
