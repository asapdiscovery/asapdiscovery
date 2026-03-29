#!/usr/bin/env python
"""Resolve internal asapdiscovery package dependencies from pyproject.toml files.

Reads the dependency graph declared in each subpackage's pyproject.toml and
outputs an install order for a given package (including itself and all
transitive internal dependencies). Handles cycles gracefully — packages
in a cycle are co-installed since order doesn't matter with --no-deps.

Usage:
    python resolve_deps.py alchemy     # prints: data docking alchemy
    python resolve_deps.py all         # prints all packages in install order
    python resolve_deps.py --graph     # prints the full dependency graph
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python <3.11 fallback

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PREFIX = "asapdiscovery-"


def discover_packages() -> dict[str, list[str]]:
    """Read all pyproject.toml files and build {pkg: [dep, ...]} graph."""
    graph: dict[str, list[str]] = {}
    for toml_path in sorted(REPO_ROOT.glob(f"{PREFIX}*/pyproject.toml")):
        data = tomllib.loads(toml_path.read_text())
        name = data["project"]["name"].removeprefix(PREFIX)
        deps = [
            d.removeprefix(PREFIX)
            for d in data["project"].get("dependencies", [])
            if d.startswith(PREFIX)
        ]
        graph[name] = deps
    return graph


def _collect_reachable(package: str, graph: dict[str, list[str]]) -> set[str]:
    """Collect all packages reachable from `package` (including itself)."""
    visited: set[str] = set()

    def walk(pkg: str) -> None:
        if pkg in visited:
            return
        if pkg not in graph:
            print(f"error: unknown package '{pkg}'", file=sys.stderr)
            print(f"available: {', '.join(sorted(graph))}", file=sys.stderr)
            sys.exit(1)
        visited.add(pkg)
        for dep in graph[pkg]:
            walk(dep)

    walk(package)
    return visited


def _topo_sort_with_cycles(subgraph: dict[str, list[str]]) -> list[str]:
    """Topological sort that handles cycles via Kahn's algorithm.

    Packages with no unresolved deps are emitted first (install order).
    Cycle members are emitted last (order among them is arbitrary but
    stable, since pip install --no-deps makes order irrelevant).
    """
    # in_degree[pkg] = number of its deps that are in the subgraph
    # (i.e. number of prerequisites not yet installed)
    in_degree: dict[str, int] = {}
    for pkg, deps in subgraph.items():
        in_degree[pkg] = sum(1 for d in deps if d in subgraph)

    # Kahn's algorithm — emit packages whose deps are all resolved
    queue = sorted(pkg for pkg, deg in in_degree.items() if deg == 0)
    order: list[str] = []
    while queue:
        pkg = queue.pop(0)
        order.append(pkg)
        # This prerequisite is now installed; decrease in-degree of dependents
        for other, deps in subgraph.items():
            if other not in in_degree:
                continue
            if pkg in deps:
                in_degree[other] -= 1
                if in_degree[other] == 0:
                    queue.append(other)
                    queue.sort()

    # Any remaining packages are in cycles — append in sorted order
    remaining = sorted(pkg for pkg in subgraph if pkg not in order)
    if remaining:
        print(
            f"note: cycle detected among: {', '.join(remaining)}",
            file=sys.stderr,
        )
    order.extend(remaining)
    return order


def resolve(package: str, graph: dict[str, list[str]]) -> list[str]:
    """Return install order for a package and its transitive deps."""
    reachable = _collect_reachable(package, graph)
    subgraph = {pkg: [d for d in graph[pkg] if d in reachable] for pkg in reachable}
    return _topo_sort_with_cycles(subgraph)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "package",
        nargs="?",
        help="Package to resolve (short name, e.g. 'alchemy'), or 'all'",
    )
    parser.add_argument(
        "--graph", action="store_true", help="Print the full dependency graph"
    )
    args = parser.parse_args()

    graph = discover_packages()

    if args.graph:
        for pkg in sorted(graph):
            deps = graph[pkg]
            dep_str = ", ".join(deps) if deps else "(none)"
            print(f"{pkg}: {dep_str}")
        return

    if not args.package:
        parser.print_help()
        sys.exit(1)

    if args.package == "all":
        subgraph = graph
    else:
        reachable = _collect_reachable(args.package, graph)
        subgraph = {
            pkg: [d for d in graph[pkg] if d in reachable] for pkg in reachable
        }

    for pkg in _topo_sort_with_cycles(subgraph):
        print(pkg)


if __name__ == "__main__":
    main()
