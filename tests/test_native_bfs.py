# file: tests/test_native_bfs.py

import gpupath._native as native

from gpupath import CpuPathEngine
from gpupath.engine.native_cpu import NativeCpuPathEngine


from gpupath.engine.native_cpu import NativeCpuPathEngine
from gpupath.graph import CSRGraph
from gpupath.types import BfsResult


def test_native_cpu_engine_bfs_returns_python_bfs_result():
    graph = CSRGraph.from_csr(
        indptr=[0, 2, 3, 4, 4],
        indices=[1, 2, 3, 3],
    )

    engine = NativeCpuPathEngine()
    result = engine.bfs(graph, 0)

    assert isinstance(result, BfsResult)
    assert result.distances == [0, 1, 1, 2]
    assert result.predecessors[0] == -1
    assert result.predecessors[1] == 0
    assert result.predecessors[2] == 0
    assert result.predecessors[3] in (1, 2)


def test_native_cpu_engine_bfs_unreachable():
    graph = CSRGraph.from_csr(
        indptr=[0, 1, 1, 2, 2, 2],
        indices=[1, 3],
    )

    engine = NativeCpuPathEngine()
    result = engine.bfs(graph, 0)

    assert result.distances == [0, 1, -1, -1, -1]
    assert result.predecessors == [-1, 0, -1, -1, -1]


def test_native_cpu_engine_bfs_bad_source_matches_python_contract():
    graph = CSRGraph.from_csr(
        indptr=[0, 1, 1],
        indices=[1],
    )

    engine = NativeCpuPathEngine()

    try:
        engine.bfs(graph, 99)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "source 99 out of range" in str(exc)


def test_native_cpu_bfs_matches_python_cpu():
    graph = CSRGraph.from_edge_list(
        num_vertices=6,
        edges=[
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (3, 4),
        ],
        directed=True,
    )

    py_engine = CpuPathEngine()
    native_engine = NativeCpuPathEngine()

    py_result = py_engine.bfs(graph, 0)
    native_result = native_engine.bfs(graph, 0)

    assert native_result.distances == py_result.distances
    assert native_result.predecessors == py_result.predecessors
