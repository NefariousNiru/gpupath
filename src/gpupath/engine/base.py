# file: gpupath/engine/base.py

from __future__ import annotations

from abc import ABC, abstractmethod

from gpupath.graph import CSRGraph
from gpupath.types import BfsResult, SsspResult


class PathEngine(ABC):
    """Abstract base class for all path-finding engine implementations.

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
    def bfs(self, graph: CSRGraph, source: int) -> BfsResult:
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
    def sssp(self, graph: CSRGraph, source: int) -> SsspResult:
        """Run Single-Source Shortest Path (SSSP) from *source* on *graph*.

        Computes the lowest-cost distance and the predecessor vertex for every
        vertex reachable from *source*. Unreachable vertices retain
        ``UNREACHABLE_DISTANCE`` and ``NO_PREDECESSOR`` as their sentinel values.

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
    def bmssp(self, graph: CSRGraph, source: int) -> SsspResult:
        """Experimental: Run BMSSP from *source* on *graph*.

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
