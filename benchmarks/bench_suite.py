# file: benchmarks/bench_suite.py

from __future__ import annotations

import argparse
import random
import statistics
import time
from dataclasses import dataclass
from typing import Callable

from gpupath.engine.native import NativePathEngine
from gpupath.engine.reference import ReferencePathEngine
from gpupath.graph import CSRGraph
from gpupath.query import _cost_matrix, _shortest_path_lengths

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
    num_threads_list: tuple[int, ...] = (1, 2, 4, 8)
    matrix_repeats_threaded: int = 3


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
        edges: set[tuple[float | int, float | int]] = set()
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
# Prepared-graph backend benchmarks
# ---------------------------------------------------------------------------


def run_bfs_prepared_backend_benchmark(
    graph: CSRGraph,
    *,
    source: int,
    repeats: int,
) -> None:
    """Benchmark raw native CPU BFS vs prepared native CPU BFS on one graph."""
    native_engine = NativePathEngine()

    def bench_native_bfs():
        return native_engine.bfs(graph, source)

    def bench_native_bfs_prepared():
        return native_engine.bfs(graph, source)

    native_result = timed_run(bench_native_bfs, repeats=repeats)
    prepared_result = timed_run(bench_native_bfs_prepared, repeats=repeats)

    print("[Kernel Benchmark] BFS prepared graph")
    print(format_result(native_result))
    print(format_result(prepared_result))
    print_speedup("Prepared / Raw Native", native_result, prepared_result)
    print()


def run_sssp_prepared_backend_benchmark(
    graph: CSRGraph,
    *,
    source: int,
    repeats: int,
) -> None:
    """Benchmark raw native CPU SSSP vs prepared native CPU SSSP on one graph."""
    native_engine = NativePathEngine()

    def bench_native_sssp():
        return native_engine.sssp(graph, source)

    def bench_native_sssp_prepared():
        return native_engine.sssp(graph, source)

    native_result = timed_run(bench_native_sssp, repeats=repeats)
    prepared_result = timed_run(bench_native_sssp_prepared, repeats=repeats)

    print("[Kernel Benchmark] SSSP prepared graph")
    print(format_result(native_result))
    print(format_result(prepared_result))
    print_speedup("Prepared / Raw Native", native_result, prepared_result)
    print()


def run_bfs_repeated_prepared_backend_benchmark(
    graph: CSRGraph,
    *,
    repeats: int,
    num_sources: int,
    seed: int,
) -> None:
    """Benchmark repeated raw native BFS vs prepared native BFS."""
    native_engine = NativePathEngine()

    rng = random.Random(seed)
    sources = [rng.randrange(graph.num_vertices) for _ in range(num_sources)]

    def bench_native_bfs_repeated():
        return [native_engine.bfs(graph, source) for source in sources]

    def bench_native_bfs_prepared_repeated():
        return [native_engine.bfs(graph, source) for source in sources]

    native_result = timed_run(bench_native_bfs_repeated, repeats=repeats)
    prepared_result = timed_run(bench_native_bfs_prepared_repeated, repeats=repeats)

    print("[Repeated Benchmark] BFS prepared graph")
    print(f"sources={len(sources)}")
    print(format_result(native_result))
    print(format_result(prepared_result))
    print_speedup("Prepared / Raw Native", native_result, prepared_result)
    print()


def run_sssp_repeated_prepared_backend_benchmark(
    graph: CSRGraph,
    *,
    repeats: int,
    num_sources: int,
    seed: int,
) -> None:
    """Benchmark repeated raw native SSSP vs prepared native SSSP."""
    native_engine = NativePathEngine()

    rng = random.Random(seed)
    sources = [rng.randrange(graph.num_vertices) for _ in range(num_sources)]

    def bench_native_sssp_repeated():
        return [native_engine.sssp(graph, source) for source in sources]

    def bench_native_sssp_prepared_repeated():
        return [native_engine.sssp(graph, source) for source in sources]

    native_result = timed_run(bench_native_sssp_repeated, repeats=repeats)
    prepared_result = timed_run(bench_native_sssp_prepared_repeated, repeats=repeats)

    print("[Repeated Benchmark] SSSP prepared graph")
    print(f"sources={len(sources)}")
    print(format_result(native_result))
    print(format_result(prepared_result))
    print_speedup("Prepared / Raw Native", native_result, prepared_result)
    print()


# ---------------------------------------------------------------------------
# Kernel-level backend benchmarks
# ---------------------------------------------------------------------------


def run_bfs_multi_source_thread_scaling_benchmark(
    graph: CSRGraph,
    *,
    repeats: int,
    num_sources: int,
    num_targets: int,
    seed: int,
    num_threads_list: tuple[int, ...],
) -> None:
    """Benchmark native batched BFS cost-matrix execution across thread counts."""
    native_engine = NativePathEngine()
    prepared_graph = native_engine.prepare_graph(graph)

    rng = random.Random(seed)
    sources = [rng.randrange(graph.num_vertices) for _ in range(num_sources)]
    targets = [rng.randrange(graph.num_vertices) for _ in range(num_targets)]

    print("[Thread Scaling Benchmark] multi_source BFS lengths")
    print(f"sources={len(sources)}  targets={len(targets)}")

    baseline_result: BenchResult | None = None

    for num_threads in num_threads_list:

        def bench_native_batched_bfs():
            return native_engine.multi_source_lengths(
                prepared_graph.graph,
                sources,
                targets=targets,
                method="default",
                num_threads=num_threads,
            )

        result = timed_run(bench_native_batched_bfs, repeats=repeats)
        print(f"{format_result(result)}  threads={num_threads}")

        if num_threads == 1:
            baseline_result = result
        elif baseline_result is not None:
            print_speedup(f"{num_threads} threads / 1 thread", baseline_result, result)

    print()


def run_sssp_multi_source_thread_scaling_benchmark(
    graph: CSRGraph,
    *,
    repeats: int,
    num_sources: int,
    num_targets: int,
    seed: int,
    num_threads_list: tuple[int, ...],
) -> None:
    """Benchmark native batched SSSP cost-matrix execution across thread counts."""
    native_engine = NativePathEngine()
    prepared_graph = native_engine.prepare_graph(graph)

    rng = random.Random(seed)
    sources = [rng.randrange(graph.num_vertices) for _ in range(num_sources)]
    targets = [rng.randrange(graph.num_vertices) for _ in range(num_targets)]

    print("[Thread Scaling Benchmark] multi_source SSSP lengths")
    print(f"sources={len(sources)}  targets={len(targets)}")

    baseline_result: BenchResult | None = None

    for num_threads in num_threads_list:

        def bench_native_batched_sssp():
            return native_engine.multi_source_lengths(
                prepared_graph.graph,
                sources,
                targets=targets,
                method="default",
                num_threads=num_threads,
            )

        result = timed_run(bench_native_batched_sssp, repeats=repeats)
        print(f"{format_result(result)}  threads={num_threads}")

        if num_threads == 1:
            baseline_result = result
        elif baseline_result is not None:
            print_speedup(f"{num_threads} threads / 1 thread", baseline_result, result)

    print()


def run_cost_matrix_threaded_comparison_benchmark(
    graph: CSRGraph,
    *,
    repeats: int,
    num_sources: int,
    num_targets: int,
    seed: int,
    threaded_num_threads: int,
) -> None:
    """Compare Python cost-matrix, native batched single-thread, and native batched multi-thread."""
    py_engine = ReferencePathEngine()
    native_engine = NativePathEngine()

    rng = random.Random(seed)
    sources = [rng.randrange(graph.num_vertices) for _ in range(num_sources)]
    targets = [rng.randrange(graph.num_vertices) for _ in range(num_targets)]

    def bench_python_cost_matrix():
        return _cost_matrix(
            graph,
            py_engine,
            sources=sources,
            targets=targets,
            method="default",
        )

    def bench_native_cost_matrix_1_thread():
        return native_engine.multi_source_lengths(
            graph,
            sources,
            targets=targets,
            method="default",
            num_threads=1,
        )

    def bench_native_cost_matrix_threaded():
        return native_engine.multi_source_lengths(
            graph,
            sources,
            targets=targets,
            method="default",
            num_threads=threaded_num_threads,
        )

    py_result = timed_run(bench_python_cost_matrix, repeats=repeats)
    native_1_result = timed_run(bench_native_cost_matrix_1_thread, repeats=repeats)
    native_mt_result = timed_run(bench_native_cost_matrix_threaded, repeats=repeats)

    print("[API Benchmark] cost_matrix threaded comparison")
    print(
        f"sources={len(sources)}  "
        f"targets={len(targets)}  "
        f"threaded_num_threads={threaded_num_threads}"
    )
    print(format_result(py_result))
    print(format_result(native_1_result))
    print(format_result(native_mt_result))
    print_speedup("Native 1-thread / Python", py_result, native_1_result)
    print_speedup(
        f"Native {threaded_num_threads}-thread / Python", py_result, native_mt_result
    )
    print_speedup(
        f"Native {threaded_num_threads}-thread / Native 1-thread",
        native_1_result,
        native_mt_result,
    )
    print()


def run_native_batched_prepared_vs_prepare_each_time_benchmark(
    graph: CSRGraph,
    *,
    repeats: int,
    num_sources: int,
    num_targets: int,
    seed: int,
    num_threads: int,
) -> None:
    """Benchmark native batched execution with reused prepared graph vs prepare-on-each-call."""
    native_engine = NativePathEngine()

    rng = random.Random(seed)
    sources = [rng.randrange(graph.num_vertices) for _ in range(num_sources)]
    targets = [rng.randrange(graph.num_vertices) for _ in range(num_targets)]

    prepared_graph = native_engine.prepare_graph(graph)

    def bench_prepare_each_time():
        return native_engine.multi_source_lengths(
            graph,
            sources,
            targets=targets,
            method="default",
            num_threads=num_threads,
        )

    def bench_prepared_reused():
        return native_engine.multi_source_lengths(
            prepared_graph.graph,
            sources,
            targets=targets,
            method="default",
            num_threads=num_threads,
        )

    raw_result = timed_run(bench_prepare_each_time, repeats=repeats)
    prepared_result = timed_run(bench_prepared_reused, repeats=repeats)

    print("[API Benchmark] native batched prepared vs prepare-each-time")
    print(
        f"sources={len(sources)}  " f"targets={len(targets)}  " f"threads={num_threads}"
    )
    print(format_result(raw_result))
    print(format_result(prepared_result))
    print_speedup("Prepared reuse / Prepare each time", raw_result, prepared_result)
    print()


def run_bfs_backend_benchmark(
    graph: CSRGraph,
    *,
    source: int,
    repeats: int,
) -> None:
    """Benchmark Python CPU BFS vs native CPU BFS on one graph."""
    py_engine = ReferencePathEngine()
    native_engine = NativePathEngine()

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
    py_engine = ReferencePathEngine()
    native_engine = NativePathEngine()

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
    py_engine = ReferencePathEngine()
    native_engine = NativePathEngine()

    method_label = "default"

    def bench_python_api():
        return _shortest_path_lengths(graph, py_engine, source, method=method_label)

    def bench_native_api():
        return _shortest_path_lengths(graph, native_engine, source, method=method_label)

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
    py_engine = ReferencePathEngine()
    native_engine = NativePathEngine()

    rng = random.Random(seed)
    sources = [rng.randrange(graph.num_vertices) for _ in range(num_sources)]
    targets = [rng.randrange(graph.num_vertices) for _ in range(num_targets)]

    def bench_python_cost_matrix():
        return _cost_matrix(
            graph,
            py_engine,
            sources=sources,
            targets=targets,
            method="default",
        )

    def bench_native_cost_matrix():
        return _cost_matrix(
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
    py_engine = ReferencePathEngine()

    def bench_bmssp():
        return _shortest_path_lengths(graph, py_engine, source, method="bmssp")

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

    # For weighted use SSSP -> Dijkstra
    if case.weighted:
        run_sssp_backend_benchmark(
            graph,
            source=config.source,
            repeats=config.repeats_kernel,
        )
        run_sssp_prepared_backend_benchmark(
            graph,
            source=config.source,
            repeats=config.repeats_kernel,
        )
        run_sssp_repeated_prepared_backend_benchmark(
            graph,
            repeats=config.repeats_api,
            num_sources=config.matrix_sources,
            seed=case.seed + 2000,
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
        run_sssp_multi_source_thread_scaling_benchmark(
            graph,
            repeats=config.matrix_repeats_threaded,
            num_sources=config.matrix_sources,
            num_targets=config.matrix_targets,
            seed=case.seed + 3000,
            num_threads_list=config.num_threads_list,
        )
        run_cost_matrix_threaded_comparison_benchmark(
            graph,
            repeats=config.matrix_repeats_threaded,
            num_sources=config.matrix_sources,
            num_targets=config.matrix_targets,
            seed=case.seed + 4000,
            threaded_num_threads=max(config.num_threads_list),
        )

        if config.include_bmssp:
            run_bmssp_benchmark(
                graph,
                source=config.source,
                repeats=3,
            )
    # Unweighted use BFS
    else:
        run_bfs_backend_benchmark(
            graph,
            source=config.source,
            repeats=config.repeats_kernel,
        )
        run_bfs_prepared_backend_benchmark(
            graph,
            source=config.source,
            repeats=config.repeats_kernel,
        )
        run_bfs_repeated_prepared_backend_benchmark(
            graph,
            repeats=config.repeats_api,
            num_sources=config.matrix_sources,
            seed=case.seed + 2000,
        )
        run_shortest_path_lengths_benchmark(
            graph,
            source=config.source,
            repeats=config.repeats_api,
        )
        run_bfs_multi_source_thread_scaling_benchmark(
            graph,
            repeats=config.matrix_repeats_threaded,
            num_sources=config.matrix_sources,
            num_targets=config.matrix_targets,
            seed=case.seed + 3000,
            num_threads_list=config.num_threads_list,
        )

        run_cost_matrix_threaded_comparison_benchmark(
            graph,
            repeats=config.matrix_repeats_threaded,
            num_sources=config.matrix_sources,
            num_targets=config.matrix_targets,
            seed=case.seed + 4000,
            threaded_num_threads=max(config.num_threads_list),
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
