# file: gpupath/engine/reference.py

from __future__ import annotations

import heapq
import math
from collections import deque

from gpupath.engine.base import PathEngine
from gpupath.engine.bmssp import _DEFAULT_K, _DEFAULT_T, _bmssp
from gpupath.graph import CSRGraph
from gpupath.types import (
    INF_FLOAT,
    NO_PREDECESSOR,
    UNREACHABLE_DISTANCE,
    BfsResult,
    SsspResult,
)


class ReferencePathEngine(PathEngine):
    """A Python-CPU-based implementation of :class:`~gpupath.engine.base.PathEngine`.

    This backend runs entirely on the host (python) and serves as the correctness
    baseline for future accelerated backends.
    """

    def bfs(self, graph: CSRGraph, source: int) -> BfsResult:
        """Run Breadth-First Search from *source* on *graph*.

        Computes the shortest-hop distance and the predecessor vertex for
        every vertex reachable from *source*. Unreachable vertices retain
        ``UNREACHABLE_DISTANCE`` and ``NO_PREDECESSOR`` as their sentinel
        values.

        The algorithm runs in ``O(V + E)`` time and ``O(V)`` auxiliary space
        where ``V`` is the number of vertices and ``E`` is the number of
        edges in *graph*.

        Args:
            graph: The graph to traverse, stored in CSR format.
            source: The vertex from which the BFS is initiated. Must be a
                valid vertex id in ``[0, graph.num_vertices)``.

        Returns:
            A :class:`~gpupath.types.BfsResult` whose ``distances[v]`` holds
            the shortest-hop distance from *source* to vertex ``v``, and
            ``predecessors[v]`` holds the vertex that discovered ``v`` during
            the traversal. Both arrays are indexed by vertex id.

        Raises:
            ValueError: If *source* is outside ``[0, graph.num_vertices)``.
        """
        if source < 0 or source >= graph.num_vertices:
            raise ValueError(f"source {source} out of range")

        distances = [UNREACHABLE_DISTANCE] * graph.num_vertices
        predecessors = [NO_PREDECESSOR] * graph.num_vertices

        queue: deque[int] = deque()
        distances[source] = 0
        queue.append(source)

        while queue:
            u = queue.popleft()
            next_distance = distances[u] + 1
            for v in graph.neighbors(u):
                if distances[v] != UNREACHABLE_DISTANCE:
                    continue
                distances[v] = next_distance
                predecessors[v] = u
                queue.append(v)

        return BfsResult(distances=distances, predecessors=predecessors)

    def sssp(self, graph: CSRGraph, source: int) -> SsspResult:
        """Run Single-Source Shortest Path (SSSP) from *source* on *graph*.

        Computes the lowest-cost distance and predecessor vertex for every
        vertex reachable from *source* using Dijkstra's algorithm with a
        binary min-heap. Unreachable vertices retain ``INF_FLOAT`` and
        ``NO_PREDECESSOR`` as their sentinel values.

        The algorithm runs in ``O((V + E) log V)`` time and ``O(V)`` auxiliary
        space where ``V`` is the number of vertices and ``E`` is the number of
        edges in *graph*.

        See Also:
            :meth:`bmssp`: deterministic ``O(m log^(2/3) n)`` alternative for
            large sparse graphs.

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
        """
        if source < 0 or source >= graph.num_vertices:
            raise ValueError(f"source {source} out of range")

        distances = [INF_FLOAT] * graph.num_vertices
        predecessors = [NO_PREDECESSOR] * graph.num_vertices

        heap: list[tuple[float, int]] = []
        distances[source] = 0.0
        heapq.heappush(heap, (0.0, source))

        while heap:
            cur_dist, u = heapq.heappop(heap)

            # Skip stale heap entries for vertices already finalised at a
            # lower cost. This is the standard lazy-deletion pattern used
            # to avoid an explicit decrease-key operation.
            if cur_dist > distances[u]:
                continue

            for v, weight in graph.weighted_neighbors(u):
                cand = cur_dist + weight
                if cand < distances[v]:
                    distances[v] = cand
                    predecessors[v] = u
                    heapq.heappush(heap, (cand, v))

        return SsspResult(distances=distances, predecessors=predecessors)

    def bmssp(self, graph: CSRGraph, source: int) -> SsspResult:
        """BMSSP: deterministic ``O(m log^(2/3) n)`` single-source shortest paths.

        Implements the algorithm from:

            Duan, Mao, Mao, Shu, and Yin — *"Breaking the Sorting Barrier
            for Directed Single-Source Shortest Paths"*,
            arXiv:2504.17033 (2024).

        This is the first deterministic algorithm to break the classic
        ``O(m + n log n)`` bound of Dijkstra's algorithm on sparse graphs
        in the comparison-addition model.

        Args:
            graph: The graph to traverse, stored in CSR format.
            source: The source vertex id in ``[0, graph.num_vertices)``.

        Returns:
            A :class:`~gpupath.types.SsspResult` with shortest distances
            and predecessors from *source*.

        Raises:
            ValueError: If *source* is outside ``[0, graph.num_vertices)``.
        """
        if source < 0 or source >= graph.num_vertices:
            raise ValueError(f"source {source} out of range")

        n = graph.num_vertices
        distances = [INF_FLOAT] * n
        predecessors = [NO_PREDECESSOR] * n
        distances[source] = 0.0

        # Number of recursion levels: ⌈log n / T⌉
        levels = int(math.ceil(math.log(max(2, n)) / _DEFAULT_T))

        _bmssp(
            graph,
            levels,
            INF_FLOAT,
            {source},
            distances,
            predecessors,
            _DEFAULT_K,
            _DEFAULT_T,
        )

        return SsspResult(distances=distances, predecessors=predecessors)
