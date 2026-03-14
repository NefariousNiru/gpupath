# file: tests/test_engine_contract.py

from __future__ import annotations

import math

from gpupath.engine.reference import ReferencePathEngine
from gpupath.graph import CSRGraph
from gpupath.query import (
    _cost_matrix,
    _predecessors,
    _shortest_path,
    _shortest_path_lengths,
)
from gpupath.types import NO_PREDECESSOR, UNREACHABLE_DISTANCE


def _engine():
    return ReferencePathEngine()


def test_engine_contract_bfs_shapes_and_sentinels() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=5,
        edges=[(0, 1), (1, 2)],
    )
    engine = _engine()

    dist = _shortest_path_lengths(graph, engine, 0)
    preds = _predecessors(graph, engine, 0)

    assert len(dist) == graph.num_vertices
    assert len(preds) == graph.num_vertices
    assert dist[0] == 0
    assert preds[0] == NO_PREDECESSOR
    assert dist[3] == UNREACHABLE_DISTANCE
    assert preds[3] == NO_PREDECESSOR


def test_engine_contract_sssp_shapes_and_sentinels() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=5,
        edges=[(0, 1, 2.0), (1, 2, 3.0)],
    )
    engine = _engine()

    dist = _shortest_path_lengths(graph, engine, 0)
    preds = _predecessors(graph, engine, 0)

    assert len(dist) == graph.num_vertices
    assert len(preds) == graph.num_vertices
    assert dist[0] == 0.0
    assert preds[0] == NO_PREDECESSOR
    assert math.isinf(dist[4])
    assert preds[4] == NO_PREDECESSOR


def test_engine_contract_shortest_path_unweighted() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=5,
        edges=[(0, 1), (1, 2), (2, 3)],
    )
    engine = _engine()

    path = _shortest_path(graph, engine, 0, 3)
    assert path == [0, 1, 2, 3]


def test_engine_contract_shortest_path_weighted() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=5,
        edges=[(0, 1, 2.0), (1, 2, 1.5), (2, 3, 4.0)],
    )
    engine = _engine()

    path = _shortest_path(graph, engine, 0, 3)
    assert path == [0, 1, 2, 3]


def test_engine_contract_cost_matrix_ordering() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=5,
        edges=[(0, 1), (1, 2), (2, 3), (0, 4)],
    )
    engine = _engine()

    matrix = _cost_matrix(
        graph,
        engine,
        sources=[0, 1],
        targets=[2, 4, 3],
    )

    assert matrix == [
        [2, 1, 3],
        [1, UNREACHABLE_DISTANCE, 2],
    ]
