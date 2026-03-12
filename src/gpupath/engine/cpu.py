# file: gpupath/engine/cpu.py

from __future__ import annotations

from collections import deque

from gpupath.engine.base import PathEngine
from gpupath.graph import CSRGraph
from gpupath.types import NO_PREDECESSOR, UNREACHABLE_DISTANCE, BfsResult


class CpuPathEngine(PathEngine):
    """A CPU-based implementation of :class:`~gpupath.engine.base.PathEngine`.

    All algorithms run on the host using pure Python. This engine is intended
    as a reference implementation and for use in environments where a GPU is
    not available.
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
