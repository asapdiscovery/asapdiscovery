import netifaces


def get_interfaces_with_dual_ip(exclude: list[str] = []) -> list[str]:
    """
    Get a list of interfaces that have both IPv4 and IPv6 addresses.

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
