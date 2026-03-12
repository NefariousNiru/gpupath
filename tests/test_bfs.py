# file: tests/test_algorithms.py

from gpupath import (
    CSRGraph,
    CpuPathEngine,
    predecessors,
    shortest_path,
    shortest_path_lengths,
)


def test_bfs_distances() -> None:
    """BFS computes correct shortest-hop distances from the source.

    The graph is a directed binary tree rooted at vertex 0. Vertices 1 and 2
    are direct children of 0, vertex 3 is a child of 1, and vertex 4 is a
    child of 2.

    Expected distances from vertex 0:

    - 0 → 0  (source itself)
    - 0 → 1  (one hop)
    - 0 → 2  (one hop)
    - 0 → 3  (two hops via 1)
    - 0 → 4  (two hops via 2)
    """
    graph = CSRGraph.from_edge_list(
        num_vertices=5,
        edges=[(0, 1), (0, 2), (1, 3), (2, 4)],
    )
    engine = CpuPathEngine()

    assert shortest_path_lengths(graph, engine, 0) == [0, 1, 1, 2, 2]


def test_bfs_predecessors_and_path() -> None:
    """BFS predecessor array and reconstructed path are consistent.

    Uses the same binary-tree topology as :func:`test_bfs_distances`.
    Verifies two properties from a BFS rooted at vertex 0:

    - ``predecessors[0] == -1``: the source has no predecessor.
    - ``predecessors[4] == 2``: vertex 4 was discovered via vertex 2.
    - ``shortest_path(0, 4) == [0, 2, 4]``: the unique shortest path of
      length 2 passes through vertex 2.
    """
    graph = CSRGraph.from_edge_list(
        num_vertices=5,
        edges=[(0, 1), (0, 2), (1, 3), (2, 4)],
    )
    engine = CpuPathEngine()

    preds = predecessors(graph, engine, 0)
    path = shortest_path(graph, engine, 0, 4)

    assert preds[0] == -1
    assert preds[4] == 2
    assert path == [0, 2, 4]


def test_unreachable_path() -> None:
    """``shortest_path`` returns an empty list when the target is unreachable.

    Graph topology (directed)::

        0 → 1     2 → 3

    The graph consists of two disconnected components. Vertex 3 is not
    reachable from vertex 0, so ``shortest_path`` must return ``[]``
    rather than raising an exception.
    """
    graph = CSRGraph.from_edge_list(
        num_vertices=4,
        edges=[(0, 1), (2, 3)],
    )
    engine = CpuPathEngine()

    assert shortest_path(graph, engine, 0, 3) == []
