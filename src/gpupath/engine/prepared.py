# file: src/gpupath/engine/prepared.py

from __future__ import annotations

from dataclasses import dataclass

from gpupath._native import NativeCsrGraph as _NativeCsrGraph
from gpupath.graph import CSRGraph


@dataclass(frozen=True, slots=True)
class NativeCpuPreparedGraph:
    """
    Prepared native graph handle for repeated path queries.

    This wrapper preserves the original Python CSRGraph while holding a native
    CSR graph instance that can be reused across repeated BFS/SSSP calls.
    """

    graph: CSRGraph
    native_graph: _NativeCsrGraph

    @classmethod
    def from_csr_graph(cls, graph: CSRGraph) -> NativeCpuPreparedGraph:
        """
        Build a prepared native graph from a Python CSRGraph.

        Args:
            graph: Python CSR graph to prepare.

        Returns:
            NativeCpuPreparedGraph wrapping the native graph handle.
        """
        if graph.weights is None:
            native_graph = _NativeCsrGraph(
                graph.num_vertices,
                graph.indptr,
                graph.indices,
            )
        else:
            native_graph = _NativeCsrGraph(
                graph.num_vertices,
                graph.indptr,
                graph.indices,
                graph.weights,
            )
        return cls(graph=graph, native_graph=native_graph)

    @property
    def num_vertices(self) -> int:
        """Return the number of vertices in the prepared graph."""
        return self.graph.num_vertices

    @property
    def is_weighted(self) -> bool:
        """Return whether the prepared graph has explicit edge weights."""
        return self.graph.weights is not None
