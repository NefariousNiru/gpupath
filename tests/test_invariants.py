# file: tests/test_invariants.py

from __future__ import annotations

import math

from gpupath import CSRGraph
from gpupath.engine import ReferencePathEngine
from gpupath.query import _shortest_path, _shortest_path_lengths


def _edge_set_unweighted(graph: CSRGraph) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for u in range(graph.num_vertices):
        for v in graph.neighbors(u):
            edges.add((u, v))
    return edges


def _edge_weight_map(graph: CSRGraph) -> dict[tuple[int, int], float]:
    edge_weights: dict[tuple[int, int], float] = {}
    for u in range(graph.num_vertices):
        for v, w in graph.weighted_neighbors(u):
            edge_weights[(u, v)] = w
    return edge_weights


def test_unweighted_shortest_path_invariants() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=6,
        edges=[
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (4, 5),
        ],
    )
    engine = ReferencePathEngine()
    source = 0
    target = 5

    path = _shortest_path(graph, engine, source, target)
    distances = _shortest_path_lengths(graph, engine, source)
    edge_set = _edge_set_unweighted(graph)

    assert path[0] == source
    assert path[-1] == target

    for u, v in zip(path, path[1:]):
        assert (u, v) in edge_set

    assert len(path) - 1 == distances[target]


def test_weighted_shortest_path_invariants() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=5,
        edges=[
            (0, 1, 2.0),
            (0, 2, 5.0),
            (1, 2, 1.0),
            (1, 3, 2.0),
            (2, 4, 1.0),
            (3, 4, 3.0),
        ],
    )
    engine = ReferencePathEngine()
    source = 0
    target = 4

    path = _shortest_path(graph, engine, source, target)
    distances = _shortest_path_lengths(graph, engine, source)
    edge_weights = _edge_weight_map(graph)

    assert path[0] == source
    assert path[-1] == target

    total = 0.0
    for u, v in zip(path, path[1:]):
        assert (u, v) in edge_weights
        total += edge_weights[(u, v)]

    assert math.isclose(total, distances[target], rel_tol=0.0, abs_tol=1e-12)


def test_unreachable_path_is_empty_and_distance_is_sentinel() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=4,
        edges=[(0, 1), (2, 3)],
    )
    engine = ReferencePathEngine()

    path = _shortest_path(graph, engine, 0, 3)
    distances = _shortest_path_lengths(graph, engine, 0)

    assert path == []
    assert distances[3] == -1


def test_unreachable_weighted_path_is_empty_and_distance_is_inf() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=4,
        edges=[(0, 1, 1.0), (2, 3, 1.0)],
    )
    engine = ReferencePathEngine()

    path = _shortest_path(graph, engine, 0, 3)
    distances = _shortest_path_lengths(graph, engine, 0)

    assert path == []
    assert math.isinf(distances[3])
