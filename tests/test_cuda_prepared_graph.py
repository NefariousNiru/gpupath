# file: tests/test_cuda_prepared_graph.py

import pytest

from gpupath import _native
from gpupath.engine.cuda_graph import CudaGraphHandle
from gpupath.graph import CSRGraph


def test_cuda_csr_graph_smoke_unweighted() -> None:
    g = _native.CudaCsrGraph(
        4,
        [0, 2, 3, 5, 5],
        [1, 2, 2, 0, 3],
    )

    assert g.num_vertices == 4
    assert g.num_edges == 5
    assert g.is_weighted is False


def test_cuda_csr_graph_smoke_weighted() -> None:
    g = _native.CudaCsrGraph(
        4,
        [0, 2, 3, 5, 5],
        [1, 2, 2, 0, 3],
        [5.0, 6.0, 7.0, 8.0, 9.0],
    )

    assert g.num_vertices == 4
    assert g.num_edges == 5
    assert g.is_weighted is True


def test_cuda_graph_handle_from_csr_graph_unweighted() -> None:
    graph = CSRGraph(
        num_vertices=4,
        indptr=[0, 2, 3, 5, 5],
        indices=[1, 2, 2, 0, 3],
    )

    handle = CudaGraphHandle.from_csr_graph(graph)

    assert handle.graph is graph
    assert handle.num_vertices == 4
    assert handle.is_weighted is False
    assert handle.cuda_graph.num_vertices == 4
    assert handle.cuda_graph.num_edges == 5
    assert handle.cuda_graph.is_weighted is False


def test_cuda_graph_handle_from_csr_graph_weighted() -> None:
    graph = CSRGraph(
        num_vertices=4,
        indptr=[0, 2, 3, 5, 5],
        indices=[1, 2, 2, 0, 3],
        weights=[5.0, 6.0, 7.0, 8.0, 9.0],
    )

    handle = CudaGraphHandle.from_csr_graph(graph)

    assert handle.graph is graph
    assert handle.num_vertices == 4
    assert handle.is_weighted is True
    assert handle.cuda_graph.num_vertices == 4
    assert handle.cuda_graph.num_edges == 5
    assert handle.cuda_graph.is_weighted is True


def test_cuda_csr_graph_invalid_indptr_raises() -> None:
    with pytest.raises(ValueError):
        _native.CudaCsrGraph(
            4,
            [1, 2, 3, 5, 5],
            [1, 2, 2, 0, 3],
        )
