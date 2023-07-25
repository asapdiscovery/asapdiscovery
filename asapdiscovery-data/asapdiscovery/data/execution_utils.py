import netifaces
from typing import List

def get_interfaces_with_dual_ip(exclude: List[str]=[]) -> str:
    interfaces = netifaces.interfaces()
    dual_ip_interfaces = []

    for interface in interfaces:
        if interface in exclude:
            continue
        addresses = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addresses and netifaces.AF_INET6 in addresses:
            dual_ip_interfaces.append(interface)

    return dual_ip_interfaces


def estimate_n_workers(work_units: int, ratio: int=3, minimum: int=1, maximum:int=10) -> int:
    """
    Estimate the number of workers to use based on the number of work units
    and the minimum and maximum number of workers to use, 
    """
    # approx  work units per worker
    n_workers = work_units // ratio
    if n_workers < minimum:
        n_workers = minimum
    elif n_workers > maximum:
        n_workers = maximum
    return n_workers
