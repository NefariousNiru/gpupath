# file: gpupath/engine/native_cpu.py

from __future__ import annotations

import gpupath._native as _native
from gpupath.graph import CSRGraph
from gpupath.types import BfsResult


class NativeCpuPathEngine:
    """A CPU-based path engine backed by the native C++ extension.

    Delegates all computation to the compiled ``_native`` module, which
    provides the same algorithms as :class:`~gpupath.engine.cpu.CpuPathEngine`
    but implemented in C++ for improved performance.

    Use this engine as a drop-in replacement for
    :class:`~gpupath.engine.cpu.CpuPathEngine` when the native extension
    has been built.
    """

    @staticmethod
    def backend_name() -> str:
        """Return the identifier for this engine backend.

        Returns:
            The string ``"native-cpu"``.
        """
        return "native-cpu"

    @staticmethod
    def native_version() -> str:
        """Return the version string reported by the native C++ module.

        Returns:
            A human-readable version string from ``_native.version()``.
        """
        return _native.version()

    @staticmethod
    def bfs(graph: CSRGraph, source: int) -> BfsResult:
        """Run Breadth-First Search from *source* on *graph*.

        Delegates to the native C++ ``bfs_unweighted`` implementation.
        Computes the shortest-hop distance and predecessor vertex for
        every vertex reachable from *source*. Unreachable vertices retain
        ``UNREACHABLE_DISTANCE`` and ``NO_PREDECESSOR`` as their sentinel
        values.

        Args:
            graph: The graph to traverse, stored in CSR format.
            source: The vertex from which the BFS is initiated. Must be a
                valid vertex id in ``[0, graph.num_vertices)``.

        Returns:
            A :class:`~gpupath.types.BfsResult` whose ``distances[v]``
            holds the shortest-hop distance from *source* to vertex ``v``,
            and ``predecessors[v]`` holds the vertex that discovered ``v``
            during traversal. Both arrays are indexed by vertex id.

        Raises:
            ValueError: If the CSR arrays are malformed.
            IndexError: If *source* is outside ``[0, graph.num_vertices)``.
        """
        return _native.bfs_unweighted(
            graph.num_vertices,
            graph.indptr,
            graph.indices,
            source,
        )
