# file: src/gpupath/engine/cuda_graph.py

from __future__ import annotations

from dataclasses import dataclass

from gpupath._native import CudaCsrGraph as _CudaCsrGraph
from gpupath.engine._prepared_graph_mixin import (
    PreparedGraphHandleMixin,
    build_prepared_graph,
)
from gpupath.graph import CSRGraph


@dataclass(frozen=True, slots=True)
class CudaGraphHandle(PreparedGraphHandleMixin):
    """
    Prepared CUDA graph handle for repeated path queries.

    This wrapper preserves the original Python CSRGraph while holding a
    CUDA-resident CSR graph instance that can be reused across repeated
    CUDA path queries.

    Notes:
        - The original Python CSRGraph remains the source of truth for Python-side
          metadata and validation semantics.
        - The native CUDA graph is an internal prepared representation stored in
          device memory.
        - This class only prepares and holds the graph. It does not execute BFS
          or SSSP by itself.
    """

    graph: CSRGraph
    cuda_graph: _CudaCsrGraph

    @classmethod
    def from_csr_graph(cls, graph: CSRGraph) -> CudaGraphHandle:
        """
        Build a prepared CUDA graph from a Python CSRGraph.

        Args:
            graph: Python CSR graph to prepare.

        Returns:
            CudaGraphHandle wrapping the CUDA prepared graph.
        """
        return cls(
            graph=graph,
            cuda_graph=build_prepared_graph(graph, _CudaCsrGraph),
        )
