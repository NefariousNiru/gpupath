# file: tests/test_bmssp_benchmark.py

from __future__ import annotations

import random
import time

import pytest

from gpupath import CpuPathEngine, CSRGraph, shortest_path_lengths


def _make_random_sparse_weighted_graph(
    *,
    num_vertices: int,
    num_edges: int,
    seed: int = 7,
) -> CSRGraph:
    rng = random.Random(seed)
    edges: set[tuple[int, int, float]] = set()

    while len(edges) < num_edges:
        src = rng.randrange(num_vertices)
        dst = rng.randrange(num_vertices)
        if src == dst:
            continue
        weight = float(rng.randint(1, 20))
        edges.add((src, dst, weight))

    return CSRGraph.from_edge_list(
        num_vertices=num_vertices,
        edges=list(edges),
        directed=True,
    )


@pytest.mark.skip(reason="BMSSP is experimental and not yet correctness-stable")
def test_bmssp_vs_default_sssp_timing() -> None:
    graph = _make_random_sparse_weighted_graph(
        num_vertices=800000,
        num_edges=3200000,
        seed=42,
    )
    engine = CpuPathEngine()
    source = 0

    t0 = time.perf_counter()
    d_default = shortest_path_lengths(graph, engine, source, method="default")
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    d_bmssp = shortest_path_lengths(graph, engine, source, method="bmssp")
    t3 = time.perf_counter()

    assert len(d_default) == len(d_bmssp) == graph.num_vertices

    mismatches: list[tuple[int, float, float]] = []
    for v, (a, b) in enumerate(zip(d_default, d_bmssp, strict=True)):
        if a == float("inf") and b == float("inf"):
            continue
        if a != b:
            mismatches.append((v, a, b))

    if mismatches:
        print(f"\n{len(mismatches)} mismatch(es) detected:")
        for v, dijkstra_dist, bmssp_dist in mismatches[:20]:
            in_edges = [u for u in range(graph.num_vertices) if v in graph.neighbors(u)]
            print(
                f"  vertex={v:4d}  dijkstra={dijkstra_dist:8.1f}  "
                f"bmssp={bmssp_dist:8.1f}  "
                f"in_degree={len(in_edges)}  "
                f"predecessors_of_v_in_dijkstra={in_edges[:5]}"
            )
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")

        inf_where_finite = [(v, a, b) for v, a, b in mismatches if b == float("inf")]
        finite_wrong = [(v, a, b) for v, a, b in mismatches if b != float("inf")]
        print(f"\n  BMSSP returned inf where Dijkstra finite: {len(inf_where_finite)}")
        print(f"  BMSSP returned wrong finite value:         {len(finite_wrong)}")

    assert not mismatches, (
        f"{len(mismatches)} vertex distance(s) differ between Dijkstra and BMSSP "
        f"(first: vertex={mismatches[0][0]}, dijkstra={mismatches[0][1]}, "
        f"bmssp={mismatches[0][2]})"
    )

    print()
    print("BMSSP benchmark on random sparse weighted graph")
    print(f"vertices={graph.num_vertices}, edges={len(graph.indices)}, source={source}")
    print(f"default sssp time: {t1 - t0:.6f} sec")
    print(f"bmssp time:        {t3 - t2:.6f} sec")
