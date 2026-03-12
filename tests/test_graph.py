# file: tests/test_graph.py

from __future__ import annotations

import pytest

from gpupath.graph import CSRGraph


def test_graph_from_edge_list_directed() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=4,
        edges=[(0, 1), (0, 2), (2, 3)],
        directed=True,
    )

    assert graph.num_vertices == 4
    assert graph.indptr == [0, 2, 2, 3, 3]
    assert graph.indices == [1, 2, 3]
    assert graph.weights is None
    assert graph.directed is True
    assert graph.is_weighted is False


def test_graph_from_edge_list_undirected() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=3,
        edges=[(0, 1), (1, 2)],
        directed=False,
    )

    assert graph.neighbors(0) == [1]
    assert sorted(graph.neighbors(1)) == [0, 2]
    assert graph.neighbors(2) == [1]


def test_graph_from_edge_list_weighted() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=3,
        edges=[(0, 1, 2.5), (1, 2, 1.25)],
        directed=True,
    )

    assert graph.is_weighted is True
    assert graph.weights == [2.5, 1.25]
    assert graph.weighted_neighbors(0) == [(1, 2.5)]
    assert graph.weighted_neighbors(1) == [(2, 1.25)]


def test_graph_from_edge_list_mixed_weighted_defaults_unweighted_edges_to_one() -> None:
    graph = CSRGraph.from_edge_list(
        num_vertices=3,
        edges=[(0, 1), (1, 2, 4.0)],
        directed=True,
    )

    assert graph.is_weighted is True
    assert graph.weighted_neighbors(0) == [(1, 1.0)]
    assert graph.weighted_neighbors(1) == [(2, 4.0)]


def test_graph_from_csr_valid() -> None:
    graph = CSRGraph.from_csr(
        indptr=[0, 2, 2, 3],
        indices=[1, 2, 0],
        weights=[1.0, 2.0, 3.0],
        directed=True,
    )

    assert graph.num_vertices == 3
    assert graph.neighbors(0) == [1, 2]
    assert graph.weighted_neighbors(2) == [(0, 3.0)]


def test_graph_from_csr_invalid_empty_indptr() -> None:
    with pytest.raises(ValueError, match="indptr must not be empty"):
        CSRGraph.from_csr(indptr=[], indices=[])


def test_graph_from_csr_invalid_start() -> None:
    with pytest.raises(ValueError, match=r"indptr\[0\] must be 0"):
        CSRGraph.from_csr(indptr=[1, 1], indices=[])


def test_graph_from_csr_invalid_last_pointer() -> None:
    with pytest.raises(ValueError, match=r"indptr\[-1\] must equal len\(indices\)"):
        CSRGraph.from_csr(indptr=[0, 1], indices=[])


def test_graph_from_csr_invalid_non_monotonic_indptr() -> None:
    with pytest.raises(ValueError, match="indptr must be non-decreasing"):
        CSRGraph.from_csr(indptr=[0, 2, 1, 2], indices=[0, 1])


def test_graph_from_csr_invalid_index_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        CSRGraph.from_csr(indptr=[0, 1, 1], indices=[2])


def test_graph_from_csr_invalid_weight_length() -> None:
    with pytest.raises(
        ValueError, match="weights must have the same length as indices"
    ):
        CSRGraph.from_csr(indptr=[0, 1], indices=[0], weights=[])


def test_graph_from_csr_negative_weight() -> None:
    with pytest.raises(ValueError, match="negative"):
        CSRGraph.from_csr(indptr=[0, 1], indices=[0], weights=[-1.0])


def test_graph_from_edge_list_invalid_vertex() -> None:
    with pytest.raises(ValueError, match="destination 4 out of range"):
        CSRGraph.from_edge_list(num_vertices=3, edges=[(0, 4)])


def test_graph_from_edge_list_invalid_num_vertices() -> None:
    with pytest.raises(ValueError, match="num_vertices must be positive"):
        CSRGraph.from_edge_list(num_vertices=0, edges=[])


def test_neighbors_out_of_range() -> None:
    graph = CSRGraph.from_edge_list(num_vertices=3, edges=[(0, 1)])

    with pytest.raises(IndexError, match="vertex 3 out of range"):
        graph.neighbors(3)

    with pytest.raises(IndexError, match="vertex -1 out of range"):
        graph.weighted_neighbors(-1)
