# file: tests/test_native_prepared_graph.py

from __future__ import annotations

import pytest

import gpupath._native as _native
from gpupath.engine.native import NativePathEngine
from gpupath.engine.native_graph import NativeGraphHandle
from gpupath.engine.reference import ReferencePathEngine
from gpupath.graph import CSRGraph

# ---------------------------------------------------------------------------
# Graph fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def unweighted_graph() -> CSRGraph:
    """Return a small directed unweighted CSR graph."""
    return CSRGraph.from_csr(
        indptr=[0, 2, 3, 5, 5],
        indices=[1, 2, 2, 0, 3],
        directed=True,
    )


@pytest.fixture
def weighted_graph() -> CSRGraph:
    """Return a small directed weighted CSR graph."""
    return CSRGraph.from_csr(
        indptr=[0, 2, 3, 5, 5],
        indices=[1, 2, 2, 0, 3],
        weights=[1.0, 4.0, 2.0, 1.5, 3.0],
        directed=True,
    )


@pytest.fixture
def disconnected_unweighted_graph() -> CSRGraph:
    """Return a graph with unreachable vertices."""
    return CSRGraph.from_csr(
        indptr=[0, 1, 1, 2, 2],
        indices=[1, 3],
        directed=True,
    )


@pytest.fixture
def disconnected_weighted_graph() -> CSRGraph:
    """Return a weighted graph with unreachable vertices."""
    return CSRGraph.from_csr(
        indptr=[0, 1, 1, 2, 2],
        indices=[1, 3],
        weights=[2.0, 5.0],
        directed=True,
    )


# ---------------------------------------------------------------------------
# CSRGraph direct-construction validation
# ---------------------------------------------------------------------------


def test_csr_graph_direct_constructor_accepts_valid_unweighted_graph() -> None:
    graph = CSRGraph(
        num_vertices=4,
        indptr=[0, 2, 3, 5, 5],
        indices=[1, 2, 2, 0, 3],
        weights=None,
        directed=True,
    )

    assert graph.num_vertices == 4
    assert graph.indptr == [0, 2, 3, 5, 5]
    assert graph.indices == [1, 2, 2, 0, 3]
    assert graph.weights is None
    assert graph.is_weighted is False


def test_csr_graph_direct_constructor_accepts_valid_weighted_graph() -> None:
    graph = CSRGraph(
        num_vertices=4,
        indptr=[0, 2, 3, 5, 5],
        indices=[1, 2, 2, 0, 3],
        weights=[1.0, 4.0, 2.0, 1.5, 3.0],
        directed=True,
    )

    assert graph.num_vertices == 4
    assert graph.weights == [1.0, 4.0, 2.0, 1.5, 3.0]
    assert graph.is_weighted is True


def test_csr_graph_direct_constructor_rejects_bad_indptr_length() -> None:
    with pytest.raises(ValueError, match="indptr must have length num_vertices \\+ 1"):
        CSRGraph(
            num_vertices=4,
            indptr=[0, 2, 3],
            indices=[1, 2, 2],
            directed=True,
        )


def test_csr_graph_direct_constructor_rejects_bad_indptr_start() -> None:
    with pytest.raises(ValueError, match=r"indptr\[0\] must be 0"):
        CSRGraph(
            num_vertices=3,
            indptr=[1, 1, 2, 2],
            indices=[1, 2],
            directed=True,
        )


def test_csr_graph_direct_constructor_rejects_non_monotonic_indptr() -> None:
    with pytest.raises(ValueError, match="indptr must be non-decreasing"):
        CSRGraph(
            num_vertices=3,
            indptr=[0, 2, 1, 2],
            indices=[1, 2],
            directed=True,
        )


def test_csr_graph_direct_constructor_rejects_bad_indices_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        CSRGraph(
            num_vertices=3,
            indptr=[0, 1, 2, 2],
            indices=[1, 99],
            directed=True,
        )


def test_csr_graph_direct_constructor_rejects_weight_length_mismatch() -> None:
    with pytest.raises(
        ValueError, match="weights must have the same length as indices"
    ):
        CSRGraph(
            num_vertices=3,
            indptr=[0, 1, 2, 2],
            indices=[1, 2],
            weights=[1.0],
            directed=True,
        )


def test_csr_graph_direct_constructor_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="only non-negative weights are supported"):
        CSRGraph(
            num_vertices=3,
            indptr=[0, 1, 2, 2],
            indices=[1, 2],
            weights=[1.0, -2.0],
            directed=True,
        )


# ---------------------------------------------------------------------------
# Native prepared graph construction
# ---------------------------------------------------------------------------


def test_prepare_graph_returns_native_cpu_prepared_graph(
    unweighted_graph: CSRGraph,
) -> None:
    engine = NativePathEngine()

    prepared = engine.prepare_graph(unweighted_graph)

    assert isinstance(prepared, NativeGraphHandle)
    assert prepared.graph is unweighted_graph
    assert prepared.num_vertices == unweighted_graph.num_vertices
    assert prepared.is_weighted is False
    assert isinstance(prepared.native_graph, _native.NativeCsrGraph)


def test_prepare_graph_returns_weighted_native_cpu_prepared_graph(
    weighted_graph: CSRGraph,
) -> None:
    engine = NativePathEngine()

    prepared = engine.prepare_graph(weighted_graph)

    assert isinstance(prepared, NativeGraphHandle)
    assert prepared.graph is weighted_graph
    assert prepared.num_vertices == weighted_graph.num_vertices
    assert prepared.is_weighted is True
    assert isinstance(prepared.native_graph, _native.NativeCsrGraph)


def test_native_prepared_graph_unweighted_properties(
    unweighted_graph: CSRGraph,
) -> None:
    prepared = NativeGraphHandle.from_csr_graph(unweighted_graph)

    native_graph = prepared.native_graph

    assert native_graph.num_vertices == 4
    assert native_graph.num_edges == 5
    assert native_graph.is_weighted is False
    assert list(native_graph.indptr) == unweighted_graph.indptr
    assert list(native_graph.indices) == unweighted_graph.indices
    assert native_graph.weights is None


def test_native_prepared_graph_weighted_properties(
    weighted_graph: CSRGraph,
) -> None:
    prepared = NativeGraphHandle.from_csr_graph(weighted_graph)

    native_graph = prepared.native_graph

    assert native_graph.num_vertices == 4
    assert native_graph.num_edges == 5
    assert native_graph.is_weighted is True
    assert list(native_graph.indptr) == weighted_graph.indptr
    assert list(native_graph.indices) == weighted_graph.indices
    assert list(native_graph.weights) == weighted_graph.weights


# ---------------------------------------------------------------------------
# Engine parity: raw native vs prepared native vs pure Python
# ---------------------------------------------------------------------------


def test_bfs_parity_unweighted_graph(unweighted_graph: CSRGraph) -> None:
    cpu_engine = ReferencePathEngine()
    native_engine = NativePathEngine()

    cpu_result = cpu_engine.bfs(unweighted_graph, 0)
    native_raw_result = native_engine.bfs(unweighted_graph, 0)
    native_prepared_result = native_engine.bfs(unweighted_graph, 0)

    assert native_raw_result.distances == cpu_result.distances
    assert native_raw_result.predecessors == cpu_result.predecessors
    assert native_prepared_result.distances == cpu_result.distances
    assert native_prepared_result.predecessors == cpu_result.predecessors


def test_bfs_parity_disconnected_unweighted_graph(
    disconnected_unweighted_graph: CSRGraph,
) -> None:
    cpu_engine = ReferencePathEngine()
    native_engine = NativePathEngine()

    cpu_result = cpu_engine.bfs(disconnected_unweighted_graph, 0)
    native_raw_result = native_engine.bfs(disconnected_unweighted_graph, 0)
    native_prepared_result = native_engine.bfs(disconnected_unweighted_graph, 0)

    assert native_raw_result.distances == cpu_result.distances
    assert native_raw_result.predecessors == cpu_result.predecessors
    assert native_prepared_result.distances == cpu_result.distances
    assert native_prepared_result.predecessors == cpu_result.predecessors


def test_sssp_parity_weighted_graph(weighted_graph: CSRGraph) -> None:
    cpu_engine = ReferencePathEngine()
    native_engine = NativePathEngine()

    cpu_result = cpu_engine.sssp(weighted_graph, 0)
    native_raw_result = native_engine.sssp(weighted_graph, 0)
    native_prepared_result = native_engine.sssp(weighted_graph, 0)

    assert native_raw_result.distances == pytest.approx(cpu_result.distances)
    assert native_raw_result.predecessors == cpu_result.predecessors
    assert native_prepared_result.distances == pytest.approx(cpu_result.distances)
    assert native_prepared_result.predecessors == cpu_result.predecessors


def test_sssp_parity_disconnected_weighted_graph(
    disconnected_weighted_graph: CSRGraph,
) -> None:
    cpu_engine = ReferencePathEngine()
    native_engine = NativePathEngine()

    cpu_result = cpu_engine.sssp(disconnected_weighted_graph, 0)
    native_raw_result = native_engine.sssp(disconnected_weighted_graph, 0)
    native_prepared_result = native_engine.sssp(disconnected_weighted_graph, 0)

    assert native_raw_result.distances == pytest.approx(cpu_result.distances)
    assert native_raw_result.predecessors == cpu_result.predecessors
    assert native_prepared_result.distances == pytest.approx(cpu_result.distances)
    assert native_prepared_result.predecessors == cpu_result.predecessors


def test_prepared_bfs_preserves_invalid_source_contract(
    unweighted_graph: CSRGraph,
) -> None:
    engine = NativePathEngine()

    with pytest.raises(ValueError, match="source 99 out of range"):
        engine.bfs(unweighted_graph, 99)


def test_prepared_sssp_preserves_invalid_source_contract(
    weighted_graph: CSRGraph,
) -> None:
    engine = NativePathEngine()

    with pytest.raises(ValueError, match="source 99 out of range"):
        engine.sssp(weighted_graph, 99)


# ---------------------------------------------------------------------------
# Native module direct overload sanity
# ---------------------------------------------------------------------------


def test_native_module_direct_bfs_prepared_overload(
    unweighted_graph: CSRGraph,
) -> None:
    prepared = NativeGraphHandle.from_csr_graph(unweighted_graph)

    result = _native.bfs_unweighted(prepared.native_graph, 0)

    assert list(result.distances) == [0, 1, 1, 2]
    assert result.predecessors[0] == -1


def test_native_module_direct_sssp_prepared_overload(
    weighted_graph: CSRGraph,
) -> None:
    prepared = NativeGraphHandle.from_csr_graph(weighted_graph)

    result = _native.sssp(prepared.native_graph, 0)

    assert list(result.distances) == pytest.approx([0.0, 1.0, 3.0, 6.0])
    assert result.predecessors[0] == -1
    assert result.predecessors[1] == 0
    assert result.predecessors[2] == 1
    assert result.predecessors[3] == 2
