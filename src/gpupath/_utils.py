# file: gpupath/_utils.py

from __future__ import annotations

from typing import Callable, Literal, Sequence

from gpupath.engine import NativePathEngine, PathEngine, ReferencePathEngine
from gpupath.graph import CSRGraph
from gpupath.types import BfsResult, Device, SsspResult


def _resolve_engine(device: Device) -> PathEngine:
    """Resolve engine from device.

    Args:
        device:
            Execution device used for the native implementation.

            - ``Device.AUTO`` selects CUDA if a compatible GPU is available,
              otherwise CPU on C++.
            - ``Device.CPU`` forces execution on the CPU on C++.
            - ``Device.CUDA`` forces execution on a CUDA-enabled GPU on C++.
            - ``Device.REFERENCE`` forces execution on the CPU with Python.

    Notes:
        The native backend is implemented in C++ for performance.

    Returns:
        PathEngine: Resolved engine from device.
    """
    if device == Device.AUTO:
        # Make this to check if cuda is available then gpu else cpu
        # Make this selection based on the method called and benchmarks
        # For example cost matrix may do really well while other may not; so keep it dynamic based on benchmarks
        # Add to docstring the logic for deciding in AUTO
        return NativePathEngine()
    elif device == Device.CPU:
        # Force CPU for all operations
        return NativePathEngine()
    elif device == Device.CUDA:
        # Check if available else throw and force everything on CUDA
        raise NotImplementedError()
    else:
        # Use Python with CPU
        return ReferencePathEngine()


def _resolve_algorithm(
    graph: CSRGraph,
    engine: PathEngine,
    method: Literal["bmssp", "default"] = "default",
) -> Callable[[CSRGraph, int], SsspResult | BfsResult]:
    """Select the appropriate traversal algorithm.

    For weighted graphs, returns :meth:`~PathEngine.bmssp` when *method*
    is ``"bmssp"``, otherwise :meth:`~PathEngine.sssp`. For unweighted graphs,
    always returns :meth:`~PathEngine.bfs` regardless of *method*.

    The returned callable should then be invoked with ``(graph, source)``.

    Args:
        graph: The graph to traverse.
        engine: The backend engine providing traversal implementations.
        method: ``"bmssp"`` to use the BMSSP algorithm on weighted graphs,
            ``"default"`` to use Dijkstra. Ignored for unweighted graphs.

    Returns:
        Callable[[CSRGraph, int], BfsResult | SsspResult]:
            The traversal algorithm to execute.
    """
    if not graph.is_weighted:
        return engine.bfs
    if method == "bmssp":
        raise NotImplementedError(
            "bmssp is experimental; not yet implemented correctly"
        )
        # return engine.bmssp(graph, source)
    return engine.sssp


def _validate_vertices(graph: CSRGraph, vertices: Sequence[int]) -> None:
    """Helper function to validate vertices."""
    for target in vertices:
        if target < 0 or target >= graph.num_vertices:
            raise ValueError(f"target {target} out of range")
