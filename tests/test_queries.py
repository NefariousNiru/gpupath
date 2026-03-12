# file: tests/test_queries.py

from __future__ import annotations

import math

import pytest

from gpupath import CpuPathEngine, CSRGraph, cost_matrix, shortest_path_lengths
from gpupath.types import UNREACHABLE_DISTANCE


def test_shortest_path_lengths_selected_targets_unweighted() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=6,
        edges=[(0, 1), (0, 2), (1, 3), (2, 4)],
    )
    engine = CpuPathEngine()

    distances = shortest_path_lengths(graph, engine, 0, targets=[0, 3, 4, 5])

    assert distances == [0, 2, 2, UNREACHABLE_DISTANCE]


def test_shortest_path_lengths_selected_targets_weighted() -> None:
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

    distances = shortest_path_lengths(graph, engine, 0, targets=[1, 2, 4, 3])

    assert distances[0] == 2.0
    assert distances[1] == 3.0
    assert distances[2] == 4.0
    assert math.isinf(distances[3])


def test_cost_matrix_unweighted() -> None:
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
    engine = CpuPathEngine()

    matrix = cost_matrix(
        graph,
        engine,
        sources=[0, 2],
        targets=[3, 4, 5],
    )

    assert matrix == [
        [2, 2, 3],
        [UNREACHABLE_DISTANCE, 1, 2],
    ]


def test_cost_matrix_weighted() -> None:
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

    matrix = cost_matrix(
        graph,
        engine,
        sources=[0, 1],
        targets=[2, 3, 4],
    )

    assert matrix == [
        [3.0, 4.0, 4.0],
        [1.0, 2.0, 2.0],
    ]


def test_cost_matrix_empty_sources() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=3,
        edges=[(0, 1)],
    )
    engine = CpuPathEngine()

    assert cost_matrix(graph, engine, sources=[], targets=[0, 1]) == []


def test_cost_matrix_invalid_target() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=3,
        edges=[(0, 1)],
    )
    engine = CpuPathEngine()

    with pytest.raises(ValueError, match="target 5 out of range"):
        cost_matrix(graph, engine, sources=[0], targets=[5])


def test_cost_matrix_invalid_source() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=3,
        edges=[(0, 1)],
    )
    engine = CpuPathEngine()

    with pytest.raises(ValueError, match="target 4 out of range"):
        cost_matrix(graph, engine, sources=[4], targets=[1])
