import warnings
from typing import Optional

import netifaces


def get_network_interfaces_with_dual_ip(exclude: list[str] = []) -> list[str]:
    """
    Get a list of network interfaces that have both IPv4 and IPv6 addresses.

    Parameters
    ----------
    exclude : list
        List of interfaces to exclude from the list of interfaces with dual IP addresses.

    Returns
    -------
    list
        List of interfaces with both IPv4 and IPv6 addresses.
    """
    interfaces = netifaces.interfaces()
    dual_ip_interfaces = []

    for interface in interfaces:
        if interface in exclude:
            continue
        addresses = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addresses and netifaces.AF_INET6 in addresses:
            dual_ip_interfaces.append(interface)

    return dual_ip_interfaces


def guess_network_interface(exclude: list[str] = []) -> Optional[str]:
    """
    Guess a network interface to use, possibly excluding some interfaces.

    Parameters
    ----------
    exclude : list
        List of interfaces to exclude from the list of interfaces with dual IP addresses.

    Returns
    -------
    Optional[str]
        Name of the best interface to use
    """
    interfaces = get_network_interfaces_with_dual_ip(exclude=exclude)
    if not interfaces:
        raise RuntimeError("No interfaces with both IPv4 and IPv6 addresses found.")
    elif len(interfaces) > 1:
        warnings.warn(
            f"Found more than one interface: {interfaces}, using the first one"
        )
    return interfaces[0]


def estimate_n_workers(
    work_units: int, ratio: int = 3, minimum: int = 1, maximum: int = 20
) -> int:
    """
    Estimate the number of workers to use for a given number of work units.

    Parameters
    ----------
    work_units : int
        Number of work units to be processed.
    ratio : int
        Approximate ratio of work units per worker.
    minimum : int
        Minimum number of workers to use.
    maximum : int
        Maximum number of workers to use.

    Returns
    -------
    int
        Estimated number of workers to use.
    """
    n_workers = work_units // ratio
    if n_workers < minimum:
        n_workers = minimum
    elif n_workers > maximum:
        n_workers = maximum
    return n_workers


def get_platform() -> str:
    """
    Get the platform name.

    Returns
    -------
    str
        Platform name.
    """
    import platform

    return platform.system().lower()


def hyperthreading_is_enabled() -> bool:
    """
    Check if hyperthreading is enabled.

    Returns
    -------
    bool
        True if hyperthreading is enabled, False otherwise.
    """
    import psutil

    return psutil.cpu_count() != psutil.cpu_count(logical=False)
