# file: gpupath/engine/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Sequence

from gpupath.engine.native_graph import NativeGraphHandle
from gpupath.graph import CSRGraph
from gpupath.types import BfsResult, SsspResult


class PathEngine(ABC):
    """Abstract base class for all path-finding engine implementations.
    Refrain from using this directly and use public APIs.

    A :class:`PathEngine` defines the interface that every backend —
    CPU, GPU, or otherwise — must satisfy. Concrete subclasses are
    responsible for providing efficient, backend-specific implementations
    of each algorithm.

    To implement a new backend, subclass :class:`PathEngine` and override
    every abstract method::

        class MyEngine(PathEngine):
            def bfs(self, graph: CSRGraph, source: int) -> BfsResult:
                ...
    """

    @abstractmethod
    def bfs(self, graph: CSRGraph | NativeGraphHandle, source: int) -> BfsResult:
        """Run Breadth-First Search from *source* on *graph*.

        Computes the shortest-hop distance and predecessor vertex for every
        vertex reachable from *source*. Vertices that cannot be reached
        retain sentinel values defined in :mod:`gpupath.types`.

        Args:
            graph: The graph to traverse, stored in CSR format.
            source: The vertex from which the BFS is initiated. Must be a
                valid vertex id in ``[0, graph.num_vertices)``.

        Returns:
            A :class:`~gpupath.types.BfsResult` whose ``distances[v]`` holds
            the shortest-hop distance from *source* to vertex ``v``, and
            ``predecessors[v]`` holds the vertex that discovered ``v``.

        Raises:
            ValueError: If *source* is outside ``[0, graph.num_vertices)``.
            NotImplementedError: If the subclass has not overridden this
                method.
        """
        raise NotImplementedError

    @abstractmethod
    def sssp(self, graph: CSRGraph | NativeGraphHandle, source: int) -> SsspResult:
        """Run Single-Source Shortest Path (SSSP) from *source* on *graph*.

        Computes the lowest-cost distance and the predecessor vertex for every
        vertex reachable from *source*. Unreachable vertices retain
        ``INF_FLOAT`` and ``NO_PREDECESSOR`` as their sentinel values.

        Args:
            graph: The graph to traverse, stored in CSR format. Edge weights
                are read from ``graph.weights``; if the graph is unweighted
                every edge is treated as having cost ``1.0``.
            source: The vertex from which the search is initiated. Must be a
                valid vertex id in ``[0, graph.num_vertices)``.

        Returns:
            An :class:`~gpupath.types.SsspResult` whose ``distances[v]`` holds
            the lowest-cost distance from *source* to vertex ``v``, and
            ``predecessors[v]`` holds the vertex that last relaxed the edge
            into ``v``. Both arrays are indexed by vertex id.

        Raises:
            ValueError: If *source* is outside ``[0, graph.num_vertices)``.
            NotImplementedError: If the subclass has not overridden this method.
        """
        raise NotImplementedError

    # @abstractmethod
    def bmssp(self, graph: CSRGraph | NativeGraphHandle, source: int) -> SsspResult:
        """Experimental/Unimplemented: Run BMSSP from *source* on *graph*.

        Implements the deterministic ``O(m log^(2/3) n)`` algorithm from:

            Duan, Mao, Mao, Shu, and Yin — *"Breaking the Sorting Barrier
            for Directed Single-Source Shortest Paths"*,
            arXiv:2504.17033 (2024).

        Computes the lowest-cost distance and the predecessor vertex for every
        vertex reachable from *source*. Unreachable vertices retain
        ``INF_FLOAT`` and ``NO_PREDECESSOR`` as their sentinel values.

        Args:
            graph: The graph to traverse, stored in CSR format. Edge weights
                are read from ``graph.weights``; if the graph is unweighted
                every edge is treated as having cost ``1.0``.
            source: The vertex from which the search is initiated. Must be a
                valid vertex id in ``[0, graph.num_vertices)``.

        Returns:
            An :class:`~gpupath.types.SsspResult` whose ``distances[v]`` holds
            the lowest-cost distance from *source* to vertex ``v``, and
            ``predecessors[v]`` holds the vertex that last relaxed the edge
            into ``v``. Both arrays are indexed by vertex id.

        Raises:
            ValueError: If *source* is outside ``[0, graph.num_vertices)``.
            NotImplementedError: If the subclass has not overridden this method.
        """
        raise NotImplementedError

    @abstractmethod
    def multi_source_lengths(
        self,
        graph: CSRGraph | NativeGraphHandle,
        sources: Sequence[int],
        targets: Sequence[int] | None = None,
        *,
        method: Literal["bmssp", "default"] = "default",
    ) -> list[list[int]] | list[list[float]]:
        """Compute shortest-path lengths for multiple sources on *graph*.

        This method defines the backend contract used by the public
        cost-matrix-style query path. It computes one row of shortest-path
        lengths per source in *sources*, preserving source order exactly.
        When *targets* is provided, each row is filtered to those target
        vertices in the same order.

        Reference engines may implement this by repeatedly calling
        :meth:`bfs` or :meth:`sssp` in Python. Native engines may override
        this with a batched backend implementation that reduces Python
        round-trips and executes the full multi-source request internally.

        Args:
            graph: The graph to traverse, stored in CSR format.
            sources: Source vertices for which shortest-path lengths should
                be computed. Each source must be a valid vertex id in
                ``[0, graph.num_vertices)``. The returned matrix row order
                must match this input order exactly.
            targets: Optional subset of target vertices. If provided, each
                returned row contains only distances to these targets, and
                column order must match this input order exactly. If omitted,
                each row contains distances for all vertices in the graph.
            method: Algorithm selection for weighted graphs. BMSSP (Experimental) or Dijkstra

        Returns:
            A matrix of shortest-path lengths. For unweighted traversal, each
            row contains integer hop distances with ``-1`` used for
            unreachable vertices. For weighted traversal, each row contains
            floating-point path costs with ``float("inf")`` used for
            unreachable vertices.

        Raises:
            ValueError: If any source or target vertex is outside
                ``[0, graph.num_vertices)``.
            NotImplementedError: If the subclass has not overridden this
                method.
        """
        raise NotImplementedError
