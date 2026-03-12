# file: tests/test_graph.py

from gpupath.graph import CSRGraph


def test_graph_from_edge_list_directed() -> None:
    """CSR arrays are correct for a small directed graph.

    Constructs a directed graph with edges 0→1, 0→2, and 2→3 and
    verifies that the ``indptr`` and ``indices`` arrays match the
    expected CSR layout:

    - Vertex 0 has two out-edges (to 1 and 2)  → ``indptr[0:2] == [0, 2]``
    - Vertex 1 has no out-edges                → ``indptr[1:3] == [2, 2]``
    - Vertex 2 has one out-edge (to 3)         → ``indptr[2:4] == [2, 3]``
    - Vertex 3 has no out-edges                → ``indptr[3:5] == [3, 3]``
    """
    graph = CSRGraph.from_edge_list(
        num_vertices=4,
        edges=[(0, 1), (0, 2), (2, 3)],
        directed=True,
    )

    assert graph.num_vertices == 4
    assert graph.indptr == [0, 2, 2, 3, 3]
    assert graph.indices == [1, 2, 3]


def test_graph_from_edge_list_undirected() -> None:
    """Neighbour lists are symmetric for an undirected graph.

    Constructs an undirected path graph  0 — 1 — 2  and verifies that
    each edge is stored in both directions:

    - Vertex 0 is adjacent only to 1.
    - Vertex 1 is adjacent to both 0 and 2.
    - Vertex 2 is adjacent only to 1.

    The neighbour list of vertex 1 is sorted before comparison because
    :meth:`~gpupath.graph.CSRGraph.from_edge_list` does not guarantee
    neighbour ordering for undirected graphs.
    """
    graph = CSRGraph.from_edge_list(
        num_vertices=3,
        edges=[(0, 1), (1, 2)],
        directed=False,
    )

    assert graph.neighbors(0) == [1]
    assert sorted(graph.neighbors(1)) == [0, 2]
    assert graph.neighbors(2) == [1]
