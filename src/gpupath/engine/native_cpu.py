# file: gpupath/engine/native_cpu.py

from __future__ import annotations

from gpupath.engine.base import PathEngine
from gpupath.graph import CSRGraph
from gpupath.types import BfsResult

import gpupath._native as _native


class NativeCpuPathEngine(PathEngine):
    """A native CPU-backed implementation of :class:`~gpupath.engine.base.PathEngine`.

    This backend dispatches core graph traversals to the compiled C++ extension
    while preserving the same Python-facing contract as :class:`CpuPathEngine`.

    It is intended to serve as the first compiled backend and to remain
    behaviorally equivalent to the pure-Python CPU engine.
    """

    def bfs(self, graph: CSRGraph, source: int) -> BfsResult:
        """Run Breadth-First Search from *source* on *graph*.

        Computes the shortest-hop distance and predecessor vertex for every
        vertex reachable from *source* using the native C++ backend.
        Unreachable vertices retain ``-1`` in both arrays.

        Args:
            graph: The graph to traverse, stored in CSR format.
            source: The vertex from which the BFS is initiated. Must be a
                valid vertex id in ``[0, graph.num_vertices)``.

        Returns:
            A :class:`~gpupath.types.BfsResult` whose ``distances[v]`` holds
            the shortest-hop distance from *source* to vertex ``v``, and
            ``predecessors[v]`` holds the vertex that discovered ``v`` during
            the traversal.

        Raises:
            ValueError: If *source* is outside ``[0, graph.num_vertices)``.
        """
        try:
            native_result = _native.bfs_unweighted(
                graph.num_vertices,
                graph.indptr,
                graph.indices,
                source,
            )
        except IndexError as exc:
            # Preserve Python engine contract parity for invalid source.
            raise ValueError(f"source {source} out of range") from exc

        return BfsResult(
            distances=list(native_result.distances),
            predecessors=list(native_result.predecessors),
        )

    def sssp(self, graph: CSRGraph, source: int):
        raise NotImplementedError("Native CPU SSSP is not implemented yet")
