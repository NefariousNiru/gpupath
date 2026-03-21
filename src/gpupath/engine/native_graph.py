# file: src/gpupath/engine/native_graph.py

from __future__ import annotations

from dataclasses import dataclass

from gpupath._native import NativeCsrGraph as _NativeCsrGraph
from gpupath.engine._prepared_graph_mixin import (
    PreparedGraphHandleMixin,
    build_prepared_graph,
)
from gpupath.graph import CSRGraph


@dataclass(frozen=True, slots=True)
class NativeGraphHandle(PreparedGraphHandleMixin):
    """
    Prepared native graph handle for repeated path queries.

    This wrapper preserves the original Python CSRGraph while holding a native
    CSR graph instance that can be reused across repeated BFS/SSSP calls.
    """

    graph: CSRGraph
    native_graph: _NativeCsrGraph

    @classmethod
    def from_csr_graph(cls, graph: CSRGraph) -> NativeGraphHandle:
        """
        Build a prepared native graph from a Python CSRGraph.

        Args:
            graph: Python CSR graph to prepare.

        Returns:
            NativeGraphHandle wrapping the native graph handle.
        """
        return cls(
            graph=graph,
            native_graph=build_prepared_graph(graph, _NativeCsrGraph),
        )
