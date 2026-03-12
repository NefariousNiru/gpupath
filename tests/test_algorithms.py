# file: tests/test_algorithms.py

from __future__ import annotations

import math

import pytest

from gpupath import (
    CpuPathEngine,
    CSRGraph,
    predecessors,
    shortest_path,
    shortest_path_lengths,
)
from gpupath.types import INF_FLOAT, NO_PREDECESSOR, UNREACHABLE_DISTANCE


def test_bfs_distances() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=5,
        edges=[(0, 1), (0, 2), (1, 3), (2, 4)],
    )
    engine = CpuPathEngine()

    assert shortest_path_lengths(graph, engine, 0) == [0, 1, 1, 2, 2]


def test_bfs_predecessors_and_path() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=5,
        edges=[(0, 1), (0, 2), (1, 3), (2, 4)],
    )
    engine = CpuPathEngine()

    preds = predecessors(graph, engine, 0)
    path = shortest_path(graph, engine, 0, 4)

    assert preds[0] == NO_PREDECESSOR
    assert preds[4] == 2
    assert path == [0, 2, 4]


def test_unreachable_path_unweighted() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=4,
        edges=[(0, 1), (2, 3)],
    )
    engine = CpuPathEngine()

    assert shortest_path(graph, engine, 0, 3) == []
    assert shortest_path_lengths(graph, engine, 0) == [
        0,
        1,
        UNREACHABLE_DISTANCE,
        UNREACHABLE_DISTANCE,
    ]


def test_weighted_shortest_path_lengths_default_sssp() -> None:
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
    engine = CpuPathEngine()

    distances = shortest_path_lengths(graph, engine, 0)
    assert distances == [0.0, 2.0, 3.0, 4.0, 4.0]


def test_weighted_shortest_path_default_sssp() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=5,
        edges=[
            (0, 1, 2.0),
            (0, 2, 5.0),
            (1, 2, 1.0),
            (2, 4, 1.0),
        ],
    )
    engine = CpuPathEngine()

    assert shortest_path(graph, engine, 0, 4) == [0, 1, 2, 4]


def test_weighted_unreachable_default_sssp() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=4,
        edges=[
            (0, 1, 1.0),
            (2, 3, 1.0),
        ],
    )
    engine = CpuPathEngine()

    assert shortest_path(graph, engine, 0, 3) == []
    distances = shortest_path_lengths(graph, engine, 0)
    assert distances[0] == 0.0
    assert distances[1] == 1.0
    assert math.isinf(distances[2])
    assert math.isinf(distances[3])


def test_source_out_of_range_bfs() -> None:
    graph = CSRGraph.from_edge_list(num_vertices=3, edges=[(0, 1)])
    engine = CpuPathEngine()

    with pytest.raises(ValueError, match="source 3 out of range"):
        shortest_path_lengths(graph, engine, 3)


def test_target_out_of_range() -> None:
    graph = CSRGraph.from_edge_list(num_vertices=3, edges=[(0, 1)])
    engine = CpuPathEngine()

    with pytest.raises(ValueError, match="target 5 out of range"):
        shortest_path(graph, engine, 0, 5)

@pytest.mark.skip(reason="BMSSP is experimental and not yet correctness-stable")
def test_bmssp_matches_default_sssp_on_weighted_graph() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=6,
        edges=[
            (0, 1, 1.0),
            (0, 2, 4.0),
            (1, 2, 2.0),
            (1, 3, 5.0),
            (2, 3, 1.0),
            (3, 4, 3.0),
            (4, 5, 2.0),
            (2, 5, 10.0),
        ],
    )
    engine = CpuPathEngine()

    d_default = shortest_path_lengths(graph, engine, 0, method="default")
    d_bmssp = shortest_path_lengths(graph, engine, 0, method="bmssp")

    assert d_default == d_bmssp

    p_default = shortest_path(graph, engine, 0, 5, method="default")
    p_bmssp = shortest_path(graph, engine, 0, 5, method="bmssp")

    assert p_default[0] == 0 and p_default[-1] == 5
    assert p_bmssp[0] == 0 and p_bmssp[-1] == 5
    assert d_default[5] == d_bmssp[5]


def test_bmssp_on_unweighted_graph_falls_back_to_bfs() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=5,
        edges=[(0, 1), (1, 2), (2, 3), (3, 4)],
    )
    engine = CpuPathEngine()

    d_default = shortest_path_lengths(graph, engine, 0, method="default")
    d_bmssp = shortest_path_lengths(graph, engine, 0, method="bmssp")

    assert d_default == d_bmssp == [0, 1, 2, 3, 4]
