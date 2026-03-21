# file: src/gpupath/engine/_prepared_graph_mixin.py

from __future__ import annotations

from typing import TypeVar

from gpupath.graph import CSRGraph

PreparedGraphT = TypeVar("PreparedGraphT")


def build_prepared_graph(
    graph: CSRGraph,
    prepared_graph_cls: type[PreparedGraphT],
) -> PreparedGraphT:
    """
    Build a prepared backend graph from a Python CSRGraph.

    This helper centralizes the weighted/unweighted constructor branching shared
    by multiple backend handle types.
    """
    if graph.weights is None:
        return prepared_graph_cls(
            graph.num_vertices,
            graph.indptr,
            graph.indices,
        )

    return prepared_graph_cls(
        graph.num_vertices,
        graph.indptr,
        graph.indices,
        graph.weights,
    )


class PreparedGraphHandleMixin:
    """
    Shared Python-side metadata helpers for prepared graph handles.

    Subclasses are expected to define:
        - graph: CSRGraph
    """

    graph: CSRGraph

    @property
    def num_vertices(self) -> int:
        """Return the number of vertices in the prepared graph."""
        return self.graph.num_vertices

    @property
    def is_weighted(self) -> bool:
        """Return whether the prepared graph has explicit edge weights."""
        return self.graph.weights is not None
