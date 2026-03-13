# file: tests/test_native_bfs.py

import gpupath._native as native
from gpupath.engine.native_cpu import NativeCpuPathEngine


def test_native_module_imports():
    assert "bootstrap" in native.version()


def test_native_wrapper_version():
    assert "bootstrap" in NativeCpuPathEngine.native_version()


def test_native_bfs_unweighted_simple():
    # 0 -> 1, 2
    # 1 -> 3
    # 2 -> 3
    # 3 -> []
    result = native.bfs_unweighted(
        num_vertices=4,
        indptr=[0, 2, 3, 4, 4],
        indices=[1, 2, 3, 3],
        source=0,
    )

    assert result.distances == [0, 1, 1, 2]
    assert result.predecessors[0] == -1
    assert result.predecessors[1] == 0
    assert result.predecessors[2] == 0
    assert result.predecessors[3] in (1, 2)


def test_native_bfs_unreachable():
    result = native.bfs_unweighted(
        num_vertices=5,
        indptr=[0, 1, 1, 2, 2, 2],
        indices=[1, 3],
        source=0,
    )

    assert result.distances == [0, 1, -1, -1, -1]
    assert result.predecessors == [-1, 0, -1, -1, -1]
