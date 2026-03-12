# file: gpupath/query.py

from __future__ import annotations

from gpupath.engine.base import PathEngine
from gpupath.graph import CSRGraph
from gpupath.types import NO_PREDECESSOR, UNREACHABLE_DISTANCE


def shortest_path_lengths(
    graph: CSRGraph,
    engine: PathEngine,
    source: int,
) -> list[int]:
    """Return the shortest-hop distance from *source* to every vertex.

    This is a thin convenience wrapper around
    :meth:`~gpupath.engine.base.PathEngine.bfs` that extracts the
    ``distances`` array from the result.

    Args:
        graph: The graph to query, stored in CSR format.
        engine: The backend engine used to run the BFS traversal.
        source: The source vertex. Must be a valid vertex id in
            ``[0, graph.num_vertices)``.

    Returns:
        A list of length ``graph.num_vertices`` where index ``v`` holds the
        shortest-hop distance from *source* to ``v``, or
        ``UNREACHABLE_DISTANCE`` if ``v`` cannot be reached.
    """
    return engine.bfs(graph, source).distances


def predecessors(
    graph: CSRGraph,
    engine: PathEngine,
    source: int,
) -> list[int]:
    """Return the BFS predecessor of every vertex relative to *source*.

    This is a thin convenience wrapper around
    :meth:`~gpupath.engine.base.PathEngine.bfs` that extracts the
    ``predecessors`` array from the result.

    Args:
        graph: The graph to query, stored in CSR format.
        engine: The backend engine used to run the BFS traversal.
        source: The source vertex. Must be a valid vertex id in
            ``[0, graph.num_vertices)``.

    Returns:
        A list of length ``graph.num_vertices`` where index ``v`` holds the
        predecessor of ``v`` along the shortest path from *source*, or
        ``NO_PREDECESSOR`` if ``v`` is the source itself or is unreachable.
    """
    return engine.bfs(graph, source).predecessors


def shortest_path(
    graph: CSRGraph,
    engine: PathEngine,
    source: int,
    target: int,
) -> list[int]:
    """Return the shortest path from *source* to *target* as an ordered list.

    Runs a BFS from *source* and reconstructs the path to *target* by
    following the predecessor chain backwards, then reversing the result.
    Returns an empty list when *target* is not reachable from *source*.

    Args:
        graph: The graph to query, stored in CSR format.
        engine: The backend engine used to run the BFS traversal.
        source: The starting vertex. Must be a valid vertex id in
            ``[0, graph.num_vertices)``.
        target: The destination vertex. Must be a valid vertex id in
            ``[0, graph.num_vertices)``.

    Returns:
        An ordered list of vertex ids representing the shortest path from
        *source* to *target*, inclusive of both endpoints. Returns an empty
        list if *target* is unreachable from *source*.

    Raises:
        ValueError: If *target* is outside ``[0, graph.num_vertices)``.
        RuntimeError: If the predecessor chain is inconsistent, indicating
            a bug in the underlying engine implementation.
    """
    result = engine.bfs(graph, source)

    if target < 0 or target >= graph.num_vertices:
        raise ValueError(f"target {target} out of range")

    if result.distances[target] == UNREACHABLE_DISTANCE:
        return []

    path: list[int] = []
    cur = target
    while cur != source:
        path.append(cur)
        cur = result.predecessors[cur]
        if cur == NO_PREDECESSOR:
            raise RuntimeError("invalid predecessor chain detected")

    path.append(source)
    path.reverse()
    return path
