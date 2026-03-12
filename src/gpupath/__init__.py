# file: gpupath/__init__.py

from gpupath.engine.cpu import CpuPathEngine
from gpupath.graph import CSRGraph
from gpupath.query import predecessors, shortest_path, shortest_path_lengths

__all__ = [
    "CSRGraph",
    "CpuPathEngine",
    "shortest_path_lengths",
    "predecessors",
    "shortest_path",
]
"""Public API for the ``gpupath`` package.

Importing from the top-level package is the recommended way to use
``gpupath``::

    from gpupath import CSRGraph, CpuPathEngine, shortest_path

Exported names
--------------
CSRGraph
    Graph container in Compressed Sparse Row format. Use
    :meth:`~gpupath.CSRGraph.from_edge_list` or
    :meth:`~gpupath.CSRGraph.from_csr` to construct one.

CpuPathEngine
    CPU-based BFS engine. Pass an instance to the query functions below
    as the *engine* argument.

shortest_path_lengths
    Return the shortest-hop distance from a source vertex to every other
    vertex in the graph.

predecessors
    Return the BFS predecessor array for a given source vertex.

shortest_path
    Reconstruct the shortest path between a source and a target vertex as
    an ordered list of vertex ids.
"""
