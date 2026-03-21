# file: tests/test_cuda_prepared_graph.py

from gpupath import _native


def test_cuda_csr_graph_smoke() -> None:
    g = _native.CudaCsrGraph(
        4,
        [0, 2, 3, 5, 5],
        [1, 2, 2, 0, 3],
    )
    assert g.num_vertices == 4
    assert g.num_edges == 5
    assert g.is_weighted is False

    g = _native.CudaCsrGraph(
        4,
        [0, 2, 3, 5, 5],
        [1, 2, 2, 0, 3],
        [5, 6, 7, 8, 9],
    )

    assert g.num_vertices == 4
    assert g.num_edges == 5
    assert g.is_weighted is True
