# file: tests/test_native_cpu_sssp.py

from __future__ import annotations

import math
import random

import pytest

import gpupath._native as native
from gpupath.engine.native import NativePathEngine
from gpupath.engine.reference import ReferencePathEngine
from gpupath.graph import CSRGraph
from gpupath.types import SsspResult

# ---------------------------------------------------------------------------
# Raw native binding tests
# ---------------------------------------------------------------------------


def test_native_sssp_binding_weighted_simple() -> None:
    """Native ``sssp`` should compute weighted shortest paths correctly."""
    result = native.sssp(
        num_vertices=4,
        indptr=[0, 2, 3, 4, 4],
        indices=[1, 2, 3, 3],
        weights=[1.0, 4.0, 2.0, 1.0],
        source=0,
    )

    assert result.distances == [0.0, 1.0, 4.0, 3.0]
    assert result.predecessors == [-1, 0, 0, 1]


def test_native_sssp_binding_unweighted_defaults_to_unit_cost() -> None:
    """Native ``sssp`` should treat missing weights as unit-cost edges."""
    result = native.sssp(
        num_vertices=4,
        indptr=[0, 2, 3, 4, 4],
        indices=[1, 2, 3, 3],
        weights=None,
        source=0,
    )

    assert result.distances == [0.0, 1.0, 1.0, 2.0]
    assert result.predecessors[0] == -1
    assert result.predecessors[1] == 0
    assert result.predecessors[2] == 0
    assert result.predecessors[3] in (1, 2)


def test_native_sssp_binding_unreachable() -> None:
    """Unreachable vertices should retain ``inf`` and ``-1`` sentinels."""
    result = native.sssp(
        num_vertices=5,
        indptr=[0, 1, 1, 2, 2, 2],
        indices=[1, 3],
        weights=[2.0, 5.0],
        source=0,
    )

    assert result.distances[0] == 0.0
    assert result.distances[1] == 2.0
    assert math.isinf(result.distances[2])
    assert math.isinf(result.distances[3])
    assert math.isinf(result.distances[4])

    assert result.predecessors == [-1, 0, -1, -1, -1]


# ---------------------------------------------------------------------------
# Native engine contract tests
# ---------------------------------------------------------------------------


def test_native_cpu_engine_sssp_returns_python_sssp_result() -> None:
    """Native CPU engine should adapt native output into Python ``SsspResult``."""
    graph = CSRGraph.from_csr(
        indptr=[0, 2, 3, 4, 4],
        indices=[1, 2, 3, 3],
        weights=[1.0, 4.0, 2.0, 1.0],
    )

    engine = NativePathEngine()
    result = engine.sssp(graph, 0)

    assert isinstance(result, SsspResult)
    assert result.distances == [0.0, 1.0, 4.0, 3.0]
    assert result.predecessors == [-1, 0, 0, 1]


def test_native_cpu_engine_sssp_unweighted_graph() -> None:
    """Native CPU engine should treat unweighted graphs as unit-cost graphs."""
    graph = CSRGraph.from_csr(
        indptr=[0, 2, 3, 4, 4],
        indices=[1, 2, 3, 3],
    )

    engine = NativePathEngine()
    result = engine.sssp(graph, 0)

    assert result.distances == [0.0, 1.0, 1.0, 2.0]
    assert result.predecessors[0] == -1
    assert result.predecessors[1] == 0
    assert result.predecessors[2] == 0
    assert result.predecessors[3] in (1, 2)


def test_native_cpu_engine_sssp_bad_source_matches_python_contract() -> None:
    """Invalid source should raise ``ValueError`` to match Python backend."""
    graph = CSRGraph.from_csr(
        indptr=[0, 1, 1],
        indices=[1],
        weights=[1.0],
    )

    engine = NativePathEngine()

    with pytest.raises(ValueError, match="source 99 out of range"):
        engine.sssp(graph, 99)


# ---------------------------------------------------------------------------
# Backend parity tests
# ---------------------------------------------------------------------------


def test_native_cpu_sssp_matches_python_cpu_weighted() -> None:
    """Native and pure-Python CPU backends should agree on weighted SSSP."""
    graph = CSRGraph.from_edge_list(
        num_vertices=6,
        edges=[
            (0, 1, 1.0),
            (0, 2, 5.0),
            (1, 2, 2.0),
            (1, 3, 4.0),
            (2, 3, 1.0),
            (3, 4, 3.0),
        ],
        directed=True,
    )

    py_engine = ReferencePathEngine()
    native_engine = NativePathEngine()

    py_result = py_engine.sssp(graph, 0)
    native_result = native_engine.sssp(graph, 0)

    assert native_result.distances == py_result.distances
    assert native_result.predecessors == py_result.predecessors


def test_native_cpu_sssp_matches_python_cpu_unweighted() -> None:
    """Native and pure-Python CPU backends should agree on unit-cost SSSP."""
    graph = CSRGraph.from_edge_list(
        num_vertices=6,
        edges=[
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (3, 4),
        ],
        directed=True,
    )

    py_engine = ReferencePathEngine()
    native_engine = NativePathEngine()

    py_result = py_engine.sssp(graph, 0)
    native_result = native_engine.sssp(graph, 0)

    assert native_result.distances == py_result.distances
    assert native_result.predecessors == py_result.predecessors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_bfs_parity(graph: CSRGraph, source: int) -> None:
    """Assert BFS parity between Python and native CPU backends."""
    py_engine = ReferencePathEngine()
    native_engine = NativePathEngine()

    py_result = py_engine.bfs(graph, source)
    native_result = native_engine.bfs(graph, source)

    assert native_result.distances == py_result.distances
    assert native_result.predecessors == py_result.predecessors


def assert_sssp_parity(graph: CSRGraph, source: int) -> None:
    """Assert SSSP parity between Python and native CPU backends."""
    py_engine = ReferencePathEngine()
    native_engine = NativePathEngine()

    py_result = py_engine.sssp(graph, source)
    native_result = native_engine.sssp(graph, source)

    assert native_result.predecessors == py_result.predecessors
    assert len(native_result.distances) == len(py_result.distances)

    for got, expected in zip(
        native_result.distances, py_result.distances, strict=False
    ):
        if math.isinf(expected):
            assert math.isinf(got)
        else:
            assert got == expected


# ---------------------------------------------------------------------------
# Deterministic edge-case tests
# ---------------------------------------------------------------------------


def test_bfs_parity_zero_edge_graph() -> None:
    graph = CSRGraph.from_csr(
        indptr=[0, 0, 0, 0],
        indices=[],
    )
    assert_bfs_parity(graph, 0)


def test_sssp_parity_zero_edge_graph() -> None:
    graph = CSRGraph.from_csr(
        indptr=[0, 0, 0, 0],
        indices=[],
    )
    assert_sssp_parity(graph, 0)


def test_bfs_parity_disconnected_components() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=6,
        edges=[
            (0, 1),
            (1, 2),
            (3, 4),
        ],
        directed=True,
    )
    assert_bfs_parity(graph, 0)
    assert_bfs_parity(graph, 3)


def test_sssp_parity_disconnected_components() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=6,
        edges=[
            (0, 1, 2.0),
            (1, 2, 3.0),
            (3, 4, 1.5),
        ],
        directed=True,
    )
    assert_sssp_parity(graph, 0)
    assert_sssp_parity(graph, 3)


def test_bfs_parity_isolated_source() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=5,
        edges=[
            (1, 2),
            (2, 3),
        ],
        directed=True,
    )
    assert_bfs_parity(graph, 0)


def test_sssp_parity_isolated_source() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=5,
        edges=[
            (1, 2, 1.0),
            (2, 3, 2.0),
        ],
        directed=True,
    )
    assert_sssp_parity(graph, 0)


def test_bfs_parity_undirected_graph() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=5,
        edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
        ],
        directed=False,
    )
    assert_bfs_parity(graph, 0)
    assert_bfs_parity(graph, 4)


def test_sssp_parity_undirected_graph() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=5,
        edges=[
            (0, 1, 1.0),
            (1, 2, 2.0),
            (2, 3, 3.0),
            (3, 4, 4.0),
        ],
        directed=False,
    )
    assert_sssp_parity(graph, 0)
    assert_sssp_parity(graph, 4)


def test_sssp_parity_unit_weight_weighted_graph() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=5,
        edges=[
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 3, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
        ],
        directed=True,
    )
    assert_sssp_parity(graph, 0)


def test_bfs_parity_tied_shortest_paths() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=5,
        edges=[
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (3, 4),
        ],
        directed=True,
    )
    assert_bfs_parity(graph, 0)


def test_sssp_parity_tied_shortest_paths() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=5,
        edges=[
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 3, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
        ],
        directed=True,
    )
    assert_sssp_parity(graph, 0)


# ---------------------------------------------------------------------------
# Randomized parity tests
# ---------------------------------------------------------------------------


def make_random_unweighted_graph(
    *,
    num_vertices: int,
    num_edges: int,
    seed: int,
) -> CSRGraph:
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()

    while len(edges) < num_edges:
        src = rng.randrange(num_vertices)
        dst = rng.randrange(num_vertices)
        if src == dst:
            continue
        edges.add((src, dst))

    return CSRGraph.from_edge_list(
        num_vertices=num_vertices,
        edges=list(edges),
        directed=True,
    )


def make_random_weighted_graph(
    *,
    num_vertices: int,
    num_edges: int,
    seed: int,
) -> CSRGraph:
    rng = random.Random(seed)
    edges: set[tuple[int, int, float]] = set()

    while len(edges) < num_edges:
        src = rng.randrange(num_vertices)
        dst = rng.randrange(num_vertices)
        if src == dst:
            continue
        weight = float(rng.randint(1, 20))
        edges.add((src, dst, weight))

    return CSRGraph.from_edge_list(
        num_vertices=num_vertices,
        edges=list(edges),
        directed=True,
    )


def test_bfs_parity_randomized_small_graphs() -> None:
    for seed in range(10):
        graph = make_random_unweighted_graph(
            num_vertices=12,
            num_edges=30,
            seed=seed,
        )
        for source in (0, 3, 7):
            assert_bfs_parity(graph, source)


def test_sssp_parity_randomized_small_graphs() -> None:
    for seed in range(10):
        graph = make_random_weighted_graph(
            num_vertices=12,
            num_edges=30,
            seed=seed,
        )
        for source in (0, 3, 7):
            assert_sssp_parity(graph, source)
