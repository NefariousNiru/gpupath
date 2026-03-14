"""
file: gpupath/__init__.py

gpupath public API.

Recommended usage:

    from gpupath import CSRGraph, CpuPathEngine, shortest_path

Primary exports:
- CSRGraph: graph container in Compressed Sparse Row format
- CpuPathEngine: CPU reference backend implementing BFS and SSSP
- shortest_path_lengths: shortest distances from a source
- predecessors: predecessor array from a source
- shortest_path: exact path reconstruction from source to target
- cost_matrix: batched source-target shortest path distances
"""

from gpupath.engine.base import PathEngine
from gpupath.engine.cpu import CpuPathEngine
from gpupath.engine.native_cpu import NativeCpuPathEngine
from gpupath.graph import CSRGraph
from gpupath.query import (
    cost_matrix,
    predecessors,
    shortest_path,
    shortest_path_lengths,
)

__all__ = [
    "CSRGraph",
    "PathEngine",
    "CpuPathEngine",
    "NativeCpuPathEngine",
    "shortest_path_lengths",
    "predecessors",
    "shortest_path",
    "cost_matrix",
]
