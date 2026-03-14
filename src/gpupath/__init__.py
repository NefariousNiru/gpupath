"""
file: gpupath/__init__.py

gpupath public API.

Usage:
    from gpupath import CSRGraph, shortest_path, predecessors, shortest_path, cost_matrix,

Primary exports:
- CSRGraph: graph container in Compressed Sparse Row format
- shortest_path_lengths: shortest distances from a source
- predecessors: predecessor array from a source
- shortest_path: exact path reconstruction from source to target
- cost_matrix: batched source-target shortest path distances
"""

from gpupath.graph import CSRGraph
from gpupath.query import (
    cost_matrix,
    predecessors,
    shortest_path,
    shortest_path_lengths,
)

__all__ = [
    "CSRGraph",
    "shortest_path_lengths",
    "predecessors",
    "shortest_path",
    "cost_matrix",
]
