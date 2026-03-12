# file: gpupath/query.py

from __future__ import annotations

from typing import Literal

from gpupath.engine.base import PathEngine
from gpupath.graph import CSRGraph
from gpupath.types import (
    INF_FLOAT,
    NO_PREDECESSOR,
    UNREACHABLE_DISTANCE,
    BfsResult,
    SsspResult,
)


def _resolve(
    graph: CSRGraph,
    engine: PathEngine,
    source: int,
    method: Literal["bmssp", "default"] = "default",
) -> SsspResult | BfsResult:
    """Select and run the appropriate traversal algorithm.

    For weighted graphs, dispatches to :meth:`~PathEngine.bmssp` when
    *method* is ``"bmssp"``, otherwise to :meth:`~PathEngine.sssp`.
    For unweighted graphs, always dispatches to :meth:`~PathEngine.bfs`
    regardless of *method*.

    Args:
        graph: The graph to traverse.
        engine: The backend engine to use.
        source: The source vertex id.
        method: ``"bmssp"`` to use the BMSSP algorithm on weighted graphs,
            ``"default"`` to use Dijkstra. Ignored for unweighted graphs.

    Returns:
        A :class:`~gpupath.types.BfsResult` or
        :class:`~gpupath.types.SsspResult` depending on the graph and method.
    """
    if not graph.is_weighted:
        return engine.bfs(graph, source)
    if method == "bmssp":
        raise NotImplementedError("bmssp is experimental; not yet implemented correctly")
        # return engine.bmssp(graph, source)
    return engine.sssp(graph, source)


def shortest_path_lengths(
    graph: CSRGraph,
    engine: PathEngine,
    source: int,
    method: Literal["bmssp", "default"] = "default",
) -> list[int] | list[float]:
    """Return the shortest distance from *source* to every vertex.

    Automatically selects BFS for unweighted graphs and SSSP for weighted
    ones. On weighted graphs, pass ``method="bmssp"`` to use the
    deterministic ``O(m log^(2/3) n)`` BMSSP algorithm instead of
    Dijkstra.

    Args:
        graph: The graph to query, stored in CSR format.
        engine: The backend engine used to run the traversal.
        source: The source vertex. Must be a valid vertex id in
            ``[0, graph.num_vertices)``.
        method: SSSP algorithm to use on weighted graphs. Either
            ``"default"`` (Dijkstra) or ``"bmssp"``. Ignored for
            unweighted graphs.

    Returns:
        A list of length ``graph.num_vertices`` where index ``v`` holds the
        shortest distance from *source* to ``v``. Unreachable vertices
        hold ``UNREACHABLE_DISTANCE`` (unweighted) or ``INF_FLOAT``
        (weighted).
    """
    return _resolve(graph, engine, source, method).distances


def predecessors(
    graph: CSRGraph,
    engine: PathEngine,
    source: int,
    method: Literal["bmssp", "default"] = "default",
) -> list[int]:
    """Return the predecessor of every vertex along the shortest path from *source*.

    Automatically selects BFS for unweighted graphs and SSSP for weighted
    ones. On weighted graphs, pass ``method="bmssp"`` to use the
    deterministic ``O(m log^(2/3) n)`` BMSSP algorithm instead of
    Dijkstra.

    Args:
        graph: The graph to query, stored in CSR format.
        engine: The backend engine used to run the traversal.
        source: The source vertex. Must be a valid vertex id in
            ``[0, graph.num_vertices)``.
        method: SSSP algorithm to use on weighted graphs. Either
            ``"default"`` (Dijkstra) or ``"bmssp"``. Ignored for
            unweighted graphs.

    Returns:
        A list of length ``graph.num_vertices`` where index ``v`` holds the
        predecessor of ``v`` along the shortest path from *source*, or
        ``NO_PREDECESSOR`` if ``v`` is the source itself or is unreachable.
    """
    return _resolve(graph, engine, source, method).predecessors


def shortest_path(
    graph: CSRGraph,
    engine: PathEngine,
    source: int,
    target: int,
    method: Literal["bmssp", "default"] = "default",
) -> list[int]:
    """Return the shortest path from *source* to *target* as an ordered list.

    Automatically selects BFS for unweighted graphs and SSSP for weighted
    ones. Reconstructs the path by following the predecessor chain
    backwards from *target* to *source*, then reversing the result.
    Returns an empty list when *target* is not reachable from *source*.

    Args:
        graph: The graph to query, stored in CSR format.
        engine: The backend engine used to run the traversal.
        source: The starting vertex. Must be a valid vertex id in
            ``[0, graph.num_vertices)``.
        target: The destination vertex. Must be a valid vertex id in
            ``[0, graph.num_vertices)``.
        method: SSSP algorithm to use on weighted graphs. Either
            ``"default"`` (Dijkstra) or ``"bmssp"``. Ignored for
            unweighted graphs.

    Returns:
        An ordered list of vertex ids from *source* to *target*, inclusive
        of both endpoints. Returns an empty list if *target* is unreachable.

    Raises:
        ValueError: If *target* is outside ``[0, graph.num_vertices)``.
        RuntimeError: If the predecessor chain is inconsistent, indicating
            a bug in the underlying engine implementation.
    """
    if target < 0 or target >= graph.num_vertices:
        raise ValueError(f"target {target} out of range")

    result = _resolve(graph, engine, source, method)

    sentinel = INF_FLOAT if graph.is_weighted else UNREACHABLE_DISTANCE
    if result.distances[target] == sentinel:
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
