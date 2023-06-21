import netifaces


def get_interfaces_with_dual_ip(exclude=[]):
    interfaces = netifaces.interfaces()
    dual_ip_interfaces = []

    for interface in interfaces:
        if interface in exclude:
            continue
        addresses = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addresses and netifaces.AF_INET6 in addresses:
            dual_ip_interfaces.append(interface)

    return dual_ip_interfaces
