# file: benchmarks/bench_suite.py

from __future__ import annotations

import argparse
import random
import statistics
import time
from dataclasses import dataclass
from typing import Callable

from gpupath import cost_matrix, shortest_path_lengths
from gpupath.engine.cpu import CpuPathEngine
from gpupath.engine.native_cpu import NativeCpuPathEngine
from gpupath.graph import CSRGraph

# ---------------------------------------------------------------------------
# Benchmark result model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BenchResult:
    """Summary container for repeated benchmark timings."""

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


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GraphCase:
    """Graph workload definition used throughout the benchmark suite."""

    name: str
    num_vertices: int
    num_edges: int
    weighted: bool
    seed: int


@dataclass(slots=True)
class SuiteConfig:
    """Top-level benchmark configuration."""

    repeats_kernel: int = 5
    repeats_api: int = 3
    source: int = 0
    matrix_sources: int = 64
    matrix_targets: int = 128
    include_bmssp: bool = True


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------


def make_random_graph(case: GraphCase) -> CSRGraph:
    """Build a random directed graph for benchmark timing.

    The generated graph is intended for reproducible backend timing rather
    than algorithm validation. Self-loops are skipped. Duplicate edges are
    removed to keep edge counts stable and graph construction honest.

    Args:
        case: Graph workload definition.

    Returns:
        A validated :class:`CSRGraph`.
    """
    rng = random.Random(case.seed)

    if case.weighted:
        edges: set[tuple[int, int, float]] = set()
        while len(edges) < case.num_edges:
            src = rng.randrange(case.num_vertices)
            dst = rng.randrange(case.num_vertices)
            if src == dst:
                continue
            weight = round(rng.uniform(0.1, 10.0), 6)
            edges.add((src, dst, float(weight)))
    else:
        edges = set()
        while len(edges) < case.num_edges:
            src = rng.randrange(case.num_vertices)
            dst = rng.randrange(case.num_vertices)
            if src == dst:
                continue
            edges.add((src, dst))

    return CSRGraph.from_edge_list(
        num_vertices=case.num_vertices,
        edges=list(edges),
        directed=True,
    )


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def timed_run(fn: Callable[[], object], *, repeats: int) -> BenchResult:
    """Measure repeated execution time for a benchmark callable.

    Args:
        fn: Zero-argument callable to benchmark.
        repeats: Number of timed repetitions.

    Returns:
        A :class:`BenchResult` containing all recorded times.
    """
    name = getattr(fn, "__name__", "benchmark")
    times_sec: list[float] = []

    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start

        # Prevent the benchmarked call from being optimized away in any
        # accidental future refactor and ensure the result is materialized.
        _ = result

        times_sec.append(elapsed)

    return BenchResult(name=name, times_sec=times_sec)


def format_result(result: BenchResult) -> str:
    """Render timing summary for one benchmark result."""
    return (
        f"{result.name:<32} "
        f"mean={result.mean_sec:.6f}s  "
        f"min={result.min_sec:.6f}s  "
        f"max={result.max_sec:.6f}s"
    )


def print_speedup(label: str, baseline: BenchResult, contender: BenchResult) -> None:
    """Print speedup of contender over baseline."""
    speedup = baseline.mean_sec / contender.mean_sec
    print(f"{label:<32} speedup={speedup:.2f}x")


# ---------------------------------------------------------------------------
# Kernel-level backend benchmarks
# ---------------------------------------------------------------------------


def run_bfs_backend_benchmark(
    graph: CSRGraph,
    *,
    source: int,
    repeats: int,
) -> None:
    """Benchmark Python CPU BFS vs native CPU BFS on one graph."""
    py_engine = CpuPathEngine()
    native_engine = NativeCpuPathEngine()

    def bench_python_bfs():
        return py_engine.bfs(graph, source)

    def bench_native_bfs():
        return native_engine.bfs(graph, source)

    py_result = timed_run(bench_python_bfs, repeats=repeats)
    native_result = timed_run(bench_native_bfs, repeats=repeats)

    print("[Kernel Benchmark] BFS")
    print(format_result(py_result))
    print(format_result(native_result))
    print_speedup("Native / Python", py_result, native_result)
    print()


def run_sssp_backend_benchmark(
    graph: CSRGraph,
    *,
    source: int,
    repeats: int,
) -> None:
    """Benchmark Python CPU SSSP vs native CPU SSSP on one graph."""
    py_engine = CpuPathEngine()
    native_engine = NativeCpuPathEngine()

    def bench_python_sssp():
        return py_engine.sssp(graph, source)

    def bench_native_sssp():
        return native_engine.sssp(graph, source)

    py_result = timed_run(bench_python_sssp, repeats=repeats)
    native_result = timed_run(bench_native_sssp, repeats=repeats)

    print("[Kernel Benchmark] SSSP")
    print(format_result(py_result))
    print(format_result(native_result))
    print_speedup("Native / Python", py_result, native_result)
    print()


# ---------------------------------------------------------------------------
# API-level benchmarks
# ---------------------------------------------------------------------------


def run_shortest_path_lengths_benchmark(
    graph: CSRGraph,
    *,
    source: int,
    repeats: int,
) -> None:
    """Benchmark API-level shortest_path_lengths for default traversal."""
    py_engine = CpuPathEngine()
    native_engine = NativeCpuPathEngine()

    method_label = "default"

    def bench_python_api():
        return shortest_path_lengths(graph, py_engine, source, method=method_label)

    def bench_native_api():
        return shortest_path_lengths(graph, native_engine, source, method=method_label)

    py_result = timed_run(bench_python_api, repeats=repeats)
    native_result = timed_run(bench_native_api, repeats=repeats)

    print("[API Benchmark] shortest_path_lengths")
    print(format_result(py_result))
    print(format_result(native_result))
    print_speedup("Native / Python", py_result, native_result)
    print()


def run_cost_matrix_benchmark(
    graph: CSRGraph,
    *,
    repeats: int,
    num_sources: int,
    num_targets: int,
    seed: int,
) -> None:
    """Benchmark API-level cost_matrix for Python and native backends."""
    py_engine = CpuPathEngine()
    native_engine = NativeCpuPathEngine()

    rng = random.Random(seed)
    sources = [rng.randrange(graph.num_vertices) for _ in range(num_sources)]
    targets = [rng.randrange(graph.num_vertices) for _ in range(num_targets)]

    def bench_python_cost_matrix():
        return cost_matrix(
            graph,
            py_engine,
            sources=sources,
            targets=targets,
            method="default",
        )

    def bench_native_cost_matrix():
        return cost_matrix(
            graph,
            native_engine,
            sources=sources,
            targets=targets,
            method="default",
        )

    py_result = timed_run(bench_python_cost_matrix, repeats=repeats)
    native_result = timed_run(bench_native_cost_matrix, repeats=repeats)

    print("[API Benchmark] cost_matrix")
    print(f"sources={len(sources)}  " f"targets={len(targets)}")
    print(format_result(py_result))
    print(format_result(native_result))
    print_speedup("Native / Python", py_result, native_result)
    print()


def run_bmssp_benchmark(
    graph: CSRGraph,
    *,
    source: int,
    repeats: int,
) -> None:
    """Benchmark experimental BMSSP on Python CPU backend only."""
    py_engine = CpuPathEngine()

    def bench_bmssp():
        return shortest_path_lengths(graph, py_engine, source, method="bmssp")

    print("[API Benchmark] BMSSP experimental")
    try:
        result = timed_run(bench_bmssp, repeats=repeats)
        print(format_result(result))
    except Exception as exc:
        print(f"BMSSP benchmark skipped due to runtime failure: {exc}")
    print()


# ---------------------------------------------------------------------------
# Workload suite
# ---------------------------------------------------------------------------


def build_graph_cases() -> list[GraphCase]:
    """Return the default benchmark workload suite.

    The suite intentionally spans sparse, medium-density, and denser graphs,
    plus both weighted and unweighted variants where appropriate.
    """
    return [
        GraphCase(
            name="bfs_sparse_small",
            num_vertices=5_000,
            num_edges=20_000,
            weighted=False,
            seed=42,
        ),
        GraphCase(
            name="bfs_medium",
            num_vertices=10_000,
            num_edges=80_000,
            weighted=False,
            seed=43,
        ),
        GraphCase(
            name="bfs_denseish",
            num_vertices=8_000,
            num_edges=240_000,
            weighted=False,
            seed=44,
        ),
        GraphCase(
            name="sssp_sparse",
            num_vertices=20_000,
            num_edges=120_000,
            weighted=True,
            seed=45,
        ),
        GraphCase(
            name="sssp_medium",
            num_vertices=15_000,
            num_edges=180_000,
            weighted=True,
            seed=46,
        ),
        GraphCase(
            name="sssp_denseish",
            num_vertices=10_000,
            num_edges=300_000,
            weighted=True,
            seed=47,
        ),
    ]


def run_case(case: GraphCase, config: SuiteConfig) -> None:
    """Run the relevant benchmark family for a single workload case."""
    graph = make_random_graph(case)

    print("=" * 88)
    print(f"Benchmark case: {case.name}")
    print(
        f"vertices={graph.num_vertices}  "
        f"edges={len(graph.indices)}  "
        f"weighted={graph.is_weighted}"
    )
    print("=" * 88)

    if case.weighted:
        run_sssp_backend_benchmark(
            graph,
            source=config.source,
            repeats=config.repeats_kernel,
        )
        run_shortest_path_lengths_benchmark(
            graph,
            source=config.source,
            repeats=config.repeats_api,
        )
        run_cost_matrix_benchmark(
            graph,
            repeats=config.repeats_api,
            num_sources=config.matrix_sources,
            num_targets=config.matrix_targets,
            seed=case.seed + 1000,
        )

        if config.include_bmssp:
            run_bmssp_benchmark(
                graph,
                source=config.source,
                repeats=3,
            )
    else:
        run_bfs_backend_benchmark(
            graph,
            source=config.source,
            repeats=config.repeats_kernel,
        )
        run_shortest_path_lengths_benchmark(
            graph,
            source=config.source,
            repeats=config.repeats_api,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> SuiteConfig:
    """Parse CLI arguments into a :class:`SuiteConfig`."""
    parser = argparse.ArgumentParser(
        description="Comprehensive gpupath benchmark suite."
    )
    parser.add_argument("--repeats-kernel", type=int, default=5)
    parser.add_argument("--repeats-api", type=int, default=3)
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--matrix-sources", type=int, default=64)
    parser.add_argument("--matrix-targets", type=int, default=128)
    parser.add_argument(
        "--skip-bmssp",
        action="store_true",
        help="Skip the experimental BMSSP API benchmark.",
    )

    args = parser.parse_args()

    return SuiteConfig(
        repeats_kernel=args.repeats_kernel,
        repeats_api=args.repeats_api,
        source=args.source,
        matrix_sources=args.matrix_sources,
        matrix_targets=args.matrix_targets,
        include_bmssp=not args.skip_bmssp,
    )


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full benchmark suite."""
    config = parse_args()

    print("gpupath benchmark suite")
    print(
        f"kernel_repeats={config.repeats_kernel}  "
        f"api_repeats={config.repeats_api}  "
        f"source={config.source}  "
        f"matrix_sources={config.matrix_sources}  "
        f"matrix_targets={config.matrix_targets}  "
        f"include_bmssp={config.include_bmssp}"
    )
    print()

    for case in build_graph_cases():
        run_case(case, config)


if __name__ == "__main__":
    main()
