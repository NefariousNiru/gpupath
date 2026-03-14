# file: gpupath/engine/native.py

from __future__ import annotations

from typing import Literal, Sequence

import gpupath._native as _native
from gpupath.engine.base import PathEngine
from gpupath.engine.native_graph import NativeGraphHandle
from gpupath.graph import CSRGraph
from gpupath.types import BfsResult, SsspResult


class NativePathEngine(PathEngine):
    """A native CPU-backed implementation of :class:`~gpupath.engine.base.PathEngine`.

    This backend dispatches core graph traversals to the compiled C++ extension
    while preserving the same Python-facing contract as :class:`ReferencePathEngine`.

    It is intended to serve as the first compiled backend and to remain
    behaviorally equivalent to the pure-Python CPU engine.
    """

    def bfs(self, graph: CSRGraph | NativeGraphHandle, source: int) -> BfsResult:
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
            if isinstance(graph, NativeGraphHandle):
                native_result = _native.bfs_unweighted(
                    graph.native_graph,
                    source,
                )
            else:
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

    def sssp(self, graph: CSRGraph | NativeGraphHandle, source: int) -> SsspResult:
        """Run Single-Source Shortest Path (SSSP) from *source* on *graph*.

        Computes the minimum-cost distance and predecessor vertex for every
        vertex reachable from *source* using the native C++ backend.
        Unreachable vertices retain ``inf`` in ``distances`` and ``-1`` in
        ``predecessors``.

        Args:
            graph: The graph to traverse, stored in CSR format.
            source: The vertex from which the search is initiated. Must be a
                valid vertex id in ``[0, graph.num_vertices)``.

        Returns:
            A :class:`~gpupath.types.SsspResult` whose ``distances[v]`` holds
            the minimum path cost from *source* to vertex ``v``, and
            ``predecessors[v]`` holds the vertex that last relaxed the edge
            into ``v``.

        Raises:
            ValueError: If *source* is outside ``[0, graph.num_vertices)``.
        """
        try:
            if isinstance(graph, NativeGraphHandle):
                native_result = _native.sssp(
                    graph.native_graph,
                    source,
                )
            else:
                native_result = _native.sssp(
                    graph.num_vertices,
                    graph.indptr,
                    graph.indices,
                    graph.weights,
                    source,
                )
        except IndexError as exc:
            raise ValueError(f"source {source} out of range") from exc

        return SsspResult(
            distances=list(native_result.distances),
            predecessors=list(native_result.predecessors),
        )

    def multi_source_lengths(
        self,
        graph: CSRGraph,
        sources: Sequence[int],
        targets: Sequence[int] | None = None,
        *,
        method: Literal["bmssp", "default"] = "default",
    ) -> list[list[int]] | list[list[float]]:
        """Compute shortest-path lengths for multiple sources on *graph*.

        Native backends should override this with a batched implementation
        that executes the full multi-source request inside the compiled
        layer, reducing Python round-trips relative to repeated single-source
        calls.

        Args:
            graph: The graph to traverse, stored in CSR format.
            sources: Source vertices for which shortest-path lengths should
                be computed.
            targets: Optional subset of target vertices to retain in each
                returned row.
            method: Algorithm selection for weighted graphs. BMSSP (Experimental) or Dijkstra

        Returns:
            A matrix of shortest-path lengths whose row order matches
            *sources* and whose optional column order matches *targets*.

        Raises:
            NotImplementedError: Always, until the native batched backend is
                implemented.
        """
        raise NotImplementedError(
            "Native multi_source_lengths() is not implemented yet."
        )

    @staticmethod
    def prepare_graph(graph: CSRGraph) -> NativeGraphHandle:
        """Prepare a Python CSR graph for repeated native CPU traversals.

        Args:
            graph: Graph to prepare.

        Returns:
            A prepared native graph wrapper that can be reused across repeated
            BFS and SSSP queries.
        """
        return NativeGraphHandle.from_csr_graph(graph)
