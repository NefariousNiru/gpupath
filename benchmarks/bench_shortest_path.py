# file: gpupath/benchmarks/bench_shortest_path.py

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass

from gpupath import CpuPathEngine, CSRGraph, cost_matrix, shortest_path_lengths


@dataclass(slots=True)
class BenchResult:
    name: str
    times_sec: list[float]

    @property
    def mean_sec(self) -> float:
        return statistics.mean(self.times_sec)

    @property
    def min_sec(self) -> float:
        return min(self.times_sec)

    @property
    def max_sec(self) -> float:
        return max(self.times_sec)


def make_random_unweighted_graph(
    *,
    num_vertices: int,
    num_edges: int,
    seed: int,
) -> CSRGraph:
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()

    while len(edges) < num_edges:
        src = rng.randrange(num_vertices)
        dst = rng.randrange(num_vertices)
        if src == dst:
            continue
        edges.add((src, dst))

    return CSRGraph.from_edge_list(
        num_vertices=num_vertices,
        edges=list(edges),
        directed=True,
    )


def make_random_weighted_graph(
    *,
    num_vertices: int,
    num_edges: int,
    seed: int,
    min_weight: int = 1,
    max_weight: int = 20,
) -> CSRGraph:
    rng = random.Random(seed)
    edges: set[tuple[int, int, float]] = set()

    while len(edges) < num_edges:
        src = rng.randrange(num_vertices)
        dst = rng.randrange(num_vertices)
        if src == dst:
            continue
        w = float(rng.randint(min_weight, max_weight))
        edges.add((src, dst, w))

    return CSRGraph.from_edge_list(
        num_vertices=num_vertices,
        edges=list(edges),
        directed=True,
    )


def timed_run(fn, repeats: int = 5) -> BenchResult:
    name = getattr(fn, "__name__", "benchmark")
    times: list[float] = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return BenchResult(name=name, times_sec=times)


def print_result(result: BenchResult) -> None:
    print(
        f"{result.name:<28} "
        f"mean={result.mean_sec:.6f}s  "
        f"min={result.min_sec:.6f}s  "
        f"max={result.max_sec:.6f}s"
    )


def run_unweighted_bfs_benchmark() -> None:
    graph = make_random_unweighted_graph(
        num_vertices=5_000,
        num_edges=20_000,
        seed=42,
    )
    engine = CpuPathEngine()
    source = 0

    def bench_bfs_all() -> list[int] | list[float]:
        return shortest_path_lengths(graph, engine, source)

    result = timed_run(bench_bfs_all, repeats=5)
    print("\n[Unweighted BFS]")
    print(f"vertices={graph.num_vertices}, edges={len(graph.indices)}, source={source}")
    print_result(result)


def run_weighted_sssp_benchmark() -> None:
    graph = make_random_weighted_graph(
        num_vertices=5_000,
        num_edges=20_000,
        seed=42,
    )
    engine = CpuPathEngine()
    source = 0

    def bench_sssp_default() -> list[int] | list[float]:
        return shortest_path_lengths(graph, engine, source, method="default")

    result = timed_run(bench_sssp_default, repeats=5)
    print("\n[Weighted SSSP - default]")
    print(f"vertices={graph.num_vertices}, edges={len(graph.indices)}, source={source}")
    print_result(result)


def run_weighted_bmssp_benchmark() -> None:
    graph = make_random_weighted_graph(
        num_vertices=1_000,
        num_edges=4_000,
        seed=42,
    )
    engine = CpuPathEngine()
    source = 0

    def bench_bmssp() -> list[int] | list[float]:
        return shortest_path_lengths(graph, engine, source, method="bmssp")

    print("\n[Weighted SSSP - BMSSP experimental]")
    print(f"vertices={graph.num_vertices}, edges={len(graph.indices)}, source={source}")
    try:
        result = timed_run(bench_bmssp, repeats=3)
        print_result(result)
    except Exception as exc:
        print(f"BMSSP benchmark skipped due to runtime failure: {exc}")


def run_cost_matrix_benchmark() -> None:
    graph = make_random_weighted_graph(
        num_vertices=4_000,
        num_edges=16_000,
        seed=99,
    )
    engine = CpuPathEngine()

    rng = random.Random(123)
    sources = [rng.randrange(graph.num_vertices) for _ in range(64)]
    targets = [rng.randrange(graph.num_vertices) for _ in range(128)]

    def bench_cost_matrix() -> list[list[int]] | list[list[float]]:
        return cost_matrix(
            graph, engine, sources=sources, targets=targets, method="default"
        )

    result = timed_run(bench_cost_matrix, repeats=3)
    print("\n[Cost Matrix]")
    print(
        f"vertices={graph.num_vertices}, edges={len(graph.indices)}, "
        f"sources={len(sources)}, targets={len(targets)}"
    )
    print_result(result)


def main() -> None:
    print("gpupath CPU benchmark suite")
    run_unweighted_bfs_benchmark()
    run_weighted_sssp_benchmark()
    run_weighted_bmssp_benchmark()
    run_cost_matrix_benchmark()


if __name__ == "__main__":
    main()
