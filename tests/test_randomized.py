# file: tests/test_randomized.py

from __future__ import annotations

import heapq
import random
from collections import deque

from gpupath import CpuPathEngine, CSRGraph, shortest_path_lengths


def _random_unweighted_graph(
    num_vertices: int, edge_prob: float, seed: int
) -> CSRGraph:
    rng = random.Random(seed)
    edges: list[tuple[int, int]] = []
    for u in range(num_vertices):
        for v in range(num_vertices):
            if u == v:
                continue
            if rng.random() < edge_prob:
                edges.append((u, v))
    return CSRGraph.from_edge_list(
        num_vertices=num_vertices, edges=edges, directed=True
    )


def _random_weighted_graph(num_vertices: int, edge_prob: float, seed: int) -> CSRGraph:
    rng = random.Random(seed)
    edges: list[tuple[int, int, float]] = []
    for u in range(num_vertices):
        for v in range(num_vertices):
            if u == v:
                continue
            if rng.random() < edge_prob:
                edges.append((u, v, float(rng.randint(1, 9))))
    return CSRGraph.from_edge_list(
        num_vertices=num_vertices, edges=edges, directed=True
    )


def _reference_bfs(graph: CSRGraph, source: int) -> list[int]:
    dist = [-1] * graph.num_vertices
    q: deque[int] = deque([source])
    dist[source] = 0

    while q:
        u = q.popleft()
        for v in graph.neighbors(u):
            if dist[v] != -1:
                continue
            dist[v] = dist[u] + 1
            q.append(v)
    return dist


def _reference_dijkstra(graph: CSRGraph, source: int) -> list[float]:
    dist = [float("inf")] * graph.num_vertices
    dist[source] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source)]

    while heap:
        cur_dist, u = heapq.heappop(heap)
        if cur_dist > dist[u]:
            continue
        for v, w in graph.weighted_neighbors(u):
            cand = cur_dist + w
            if cand < dist[v]:
                dist[v] = cand
                heapq.heappush(heap, (cand, v))
    return dist


def test_randomized_bfs_parity() -> None:
    engine = CpuPathEngine()

    for seed in range(20):
        graph = _random_unweighted_graph(num_vertices=12, edge_prob=0.18, seed=seed)
        for source in range(graph.num_vertices):
            got = shortest_path_lengths(graph, engine, source)
            want = _reference_bfs(graph, source)
            assert got == want


def test_randomized_weighted_sssp_parity() -> None:
    engine = CpuPathEngine()

    for seed in range(20):
        graph = _random_weighted_graph(
            num_vertices=12, edge_prob=0.18, seed=1000 + seed
        )
        for source in range(graph.num_vertices):
            got = shortest_path_lengths(graph, engine, source, method="default")
            want = _reference_dijkstra(graph, source)
            assert got == want
