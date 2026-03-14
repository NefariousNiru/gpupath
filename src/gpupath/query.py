# file: gpupath/query.py

from __future__ import annotations

from typing import Literal, Sequence

from gpupath import _utils
from gpupath.engine.base import PathEngine
from gpupath.graph import CSRGraph
from gpupath.types import (
    INF_FLOAT,
    NO_PREDECESSOR,
    UNREACHABLE_DISTANCE,
    Device,
)

# =========================================
# Public API
# =========================================


def shortest_path_lengths(
    graph: CSRGraph,
    source: int,
    device: Device = Device.AUTO,
    method: Literal["bmssp", "default"] = "default",
    targets: Sequence[int] | None = None,
) -> list[int] | list[float]:
    """Compute shortest path distances from a source vertex.

    Automatically selects the appropriate traversal algorithm based on the
    graph type:

    - **Unweighted graphs** use breadth-first search (BFS).
    - **Weighted graphs** use single-source shortest path (SSSP) with
      Dijkstra by default.

    When ``method="bmssp"`` is specified on weighted graphs, the BMSSP
    algorithm is used instead of Dijkstra (currently experimental).

    The execution backend is selected using the ``device`` parameter.

    Args:
        graph:
            The graph to query, stored in CSR format.

        source:
            The source vertex. Must be a valid vertex id in
            ``[0, graph.num_vertices)``.

        device:
            Execution device used for the native implementation.

            - ``Device.AUTO`` selects CUDA if a compatible GPU is available,
              otherwise CPU on C++.
            - ``Device.CPU`` forces execution on the CPU on C++.
            - ``Device.CUDA`` forces execution on a CUDA-enabled GPU on C++.
            - ``Device.REFERENCE``forces execution on the CPU with Python.
            The native backend is implemented in C++ for performance.

        method:
            Algorithm to use for weighted graphs.

            - ``"default"`` — Dijkstra's algorithm
            - ``"bmssp"`` — BMSSP deterministic parallel algorithm
              (currently experimental)

            Ignored for unweighted graphs.

        targets:
            Optional sequence of destination vertex ids to extract from the
            computed distance array. If ``None``, distances to all vertices
            are returned. If provided, only the requested vertices are
            returned in the same order.

    Returns:
        list[int] | list[float]:
            A list of shortest path distances.

            If ``targets`` is ``None``, the returned list has length
            ``graph.num_vertices`` where index ``v`` stores the distance from
            ``source`` to vertex ``v``.

            If ``targets`` is provided, the returned list contains only the
            distances for those vertices.

            Unreachable vertices contain ``UNREACHABLE_DISTANCE`` for
            unweighted graphs or ``INF_FLOAT`` for weighted graphs.
    """
    engine = _utils._resolve_engine(device=device)
    return _shortest_path_lengths(graph, engine, source, method=method, targets=targets)


def predecessors(
    graph: CSRGraph,
    source: int,
    device: Device = Device.AUTO,
    method: Literal["bmssp", "default"] = "default",
) -> list[int]:
    """Compute the predecessor of each vertex on the shortest path from a source.

    For unweighted graphs, breadth-first search (BFS) is used. For weighted
    graphs, single-source shortest path (SSSP) with Dijkstra is used by
    default.

    When ``method="bmssp"`` is specified on weighted graphs, the BMSSP
    algorithm is used instead of Dijkstra (currently experimental).

    The execution backend is selected using the ``device`` parameter.

    Args:
        graph:
            The graph to query, stored in CSR format.

        source:
            The source vertex. Must be a valid vertex id in
            ``[0, graph.num_vertices)``.

        device:
            Execution device used for the native implementation.

            - ``Device.AUTO`` selects CUDA if a compatible GPU is available,
              otherwise CPU on C++.
            - ``Device.CPU`` forces execution on the CPU on C++.
            - ``Device.CUDA`` forces execution on a CUDA-enabled GPU on C++.
            - ``Device.REFERENCE``forces execution on the CPU with Python.
            The native backend is implemented in C++ for performance.

        method:
            Algorithm to use for weighted graphs.

            - ``"default"`` — Dijkstra's algorithm
            - ``"bmssp"`` — BMSSP deterministic parallel algorithm
              (currently experimental)

            Ignored for unweighted graphs.

    Returns:
        list[int]:
            A list of length ``graph.num_vertices`` where index ``v`` holds
            the predecessor of ``v`` along the shortest path from ``source``.

            Vertices that are unreachable or the source vertex itself have
            value ``NO_PREDECESSOR``.
    """
    engine = _utils._resolve_engine(device=device)
    return _predecessors(graph, engine, source, method)


def shortest_path(
    graph: CSRGraph,
    source: int,
    target: int,
    device: Device = Device.AUTO,
    method: Literal["bmssp", "default"] = "default",
) -> list[int]:
    """Return the shortest path from *source* to *target* as an ordered list.

    Automatically selects BFS for unweighted graphs and SSSP for weighted
    ones. Reconstructs the path by following the predecessor chain
    backwards from *target* to *source*, then reversing the result.
    Returns an empty list when *target* is not reachable from *source*.

    Args:
        graph: The graph to query, stored in CSR format.
        device:
            Execution device used for the native implementation.

            - ``Device.AUTO`` selects CUDA if a compatible GPU is available,
              otherwise CPU on C++.
            - ``Device.CPU`` forces execution on the CPU on C++.
            - ``Device.CUDA`` forces execution on a CUDA-enabled GPU on C++.
            - ``Device.REFERENCE``forces execution on the CPU with Python.
            The native backend is implemented in C++ for performance.

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
    engine = _utils._resolve_engine(device=device)
    return _shortest_path(graph, engine, source, target, method)


def cost_matrix(
    graph: CSRGraph,
    sources: Sequence[int],
    targets: Sequence[int],
    device: Device = Device.AUTO,
    method: Literal["bmssp", "default"] = "default",
) -> list[list[int]] | list[list[float]]:
    """Compute a cost matrix of shortest distances from each source to each target.

    Runs one SSSP (or BFS for unweighted graphs) per source vertex and
    extracts the distance to each target, producing a
    ``len(sources) × len(targets)`` matrix.

    Args:
        graph: The graph to query, stored in CSR format.
        device:
            Execution device used for the native implementation.

            - ``Device.AUTO`` selects CUDA if a compatible GPU is available,
              otherwise CPU on C++.
            - ``Device.CPU`` forces execution on the CPU on C++.
            - ``Device.CUDA`` forces execution on a CUDA-enabled GPU on C++.
            - ``Device.REFERENCE``forces execution on the CPU with Python.
            The native backend is implemented in C++ for performance.

        sources: Sequence of source vertex ids. Each must be a valid vertex
            id in ``[0, graph.num_vertices)``. Returns an empty list if
            *sources* is empty.
        targets: Sequence of target vertex ids. Each must be a valid vertex
            id in ``[0, graph.num_vertices)``.
        method: SSSP algorithm to use on weighted graphs. Either
            ``"default"`` (Dijkstra) or ``"bmssp"`` (deterministic
            ``O(m log^(2/3) n)`` algorithm from arXiv:2504.17033). Ignored
            for unweighted graphs, which always use BFS.

    Returns:
        A ``len(sources) × len(targets)`` matrix where ``matrix[i][j]``
        holds the shortest distance from ``sources[i]`` to ``targets[j]``.
        Unreachable pairs hold ``UNREACHABLE_DISTANCE`` (unweighted) or
        ``INF_FLOAT`` (weighted).

    Raises:
        ValueError: If any vertex in *sources* or *targets* is outside
            ``[0, graph.num_vertices)``.
    """
    engine = _utils._resolve_engine(device=device)
    return _cost_matrix(graph, engine, sources, targets, method)


# =========================================
# Helpers
# =========================================


def _shortest_path_lengths(
    graph: CSRGraph,
    engine: PathEngine,
    source: int,
    method: Literal["bmssp", "default"] = "default",
    targets: Sequence[int] | None = None,
) -> list[int] | list[float]:
    """Internal helper to compute shortest paths using a specific engine.

    Selects the appropriate traversal algorithm based on graph type and
    delegates execution to the provided :class:`PathEngine`.

    Args:
        graph: The graph stored in CSR format.
        engine: Backend engine used to execute the traversal.
        source: Source vertex.
        method: Algorithm selection for weighted graphs.
        targets: Optional subset of vertices to extract from results.

    Returns:
        list[int] | list[float]:
            Distances from the source vertex.
    """
    result = _utils._resolve_algorithm(graph, engine, method)(graph, source)

    if targets is None:
        return result.distances

    _utils._validate_vertices(graph, targets)
    return [result.distances[t] for t in targets]


def _predecessors(
    graph: CSRGraph,
    engine: PathEngine,
    source: int,
    method: Literal["bmssp", "default"] = "default",
) -> list[int]:
    """Internal helper to compute predecessor vertices using a specific engine.

    Selects the appropriate traversal algorithm based on graph type and
    delegates execution to the provided :class:`PathEngine`.

    Args:
        graph: The graph stored in CSR format.
        engine: Backend engine used to execute the traversal.
        source: Source vertex.
        method: Algorithm selection for weighted graphs.

    Returns:
        list[int]:
            Predecessor of each vertex along the shortest path tree.
    """
    return _utils._resolve_algorithm(graph, engine, method)(graph, source).predecessors


def _shortest_path(
    graph: CSRGraph,
    engine: PathEngine,
    source: int,
    target: int,
    method: Literal["bmssp", "default"] = "default",
) -> list[int]:
    """Internal helper to compute a shortest path using a specific engine.

    Selects the appropriate traversal algorithm based on graph type and
    delegates execution to the provided :class:`PathEngine`. The path is
    reconstructed by following the predecessor chain from ``target`` back
    to ``source``.

    Args:
        graph: The graph stored in CSR format.
        engine: Backend engine used to execute the traversal.
        source: Source vertex.
        target: Destination vertex.
        method: Algorithm selection for weighted graphs.

    Returns:
        list[int]:
            Ordered list of vertex ids from ``source`` to ``target``.
            Returns an empty list if the target is unreachable.
    """

    _utils._validate_vertices(graph, [target])

    result = _utils._resolve_algorithm(graph, engine, method)(graph, source)

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


def _cost_matrix(
    graph: CSRGraph,
    engine: PathEngine,
    sources: Sequence[int],
    targets: Sequence[int],
    method: Literal["bmssp", "default"] = "default",
) -> list[list[int]] | list[list[float]]:
    """Internal helper to compute a cost matrix using a specific engine.

    Selects the appropriate traversal algorithm based on graph type and
    delegates execution to the provided :class:`PathEngine`. Runs one BFS
    (unweighted graphs) or SSSP (weighted graphs) per source vertex and
    extracts distances to the requested targets.

    Args:
        graph: The graph stored in CSR format.
        engine: Backend engine used to execute the traversal.
        sources: Sequence of source vertex ids.
        targets: Sequence of target vertex ids.
        method: Algorithm selection for weighted graphs.

    Returns:
        list[list[int]] | list[list[float]]:
            A ``len(sources) × len(targets)`` matrix where ``matrix[i][j]``
            holds the shortest distance from ``sources[i]`` to ``targets[j]``.
    """
    if not sources:
        return []

    _utils._validate_vertices(graph, sources)
    _utils._validate_vertices(graph, targets)

    return engine.multi_source_lengths(graph, sources, targets, method=method)
