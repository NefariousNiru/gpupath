# file: gpupath/engine/bmssp.py

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from gpupath.graph import CSRGraph
from gpupath.types import INF_FLOAT

# ---------------------------------------------------------------------------
# Tuning constants (mirror the Go implementation's defaults)
# ---------------------------------------------------------------------------

_DEFAULT_T: int = 3
"""Recursion depth parameter. Controls the number of levels as ⌈log n / T⌉.     
Notes:
    - Made intentional non-public with `_`
"""

_DEFAULT_K: int = 4
"""Frontier expansion factor. Controls how many vertices are explored per level.
Notes:
    - Made intentional non-public with `_`
"""


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class _LevelQueue:
    """Heap-backed approximation of the BMSSP level queue.

    Semantics that matter for correctness:
    - Insert: keep the best key per vertex, reject keys >= upper_bound.
    - Pull: remove up to ``block_size`` vertices with the smallest keys and
      return a separating bound for the remaining queue.
    - BatchPrepend: for correctness, this can reuse Insert semantics even
      though the asymptotically optimal paper structure is more specialized.

    This is correctness-oriented, not the paper's optimal block structure.
    """

    upper_bound: float = INF_FLOAT
    block_size: int = 1
    _heap: list[tuple[float, int]] = field(default_factory=list)
    _best: dict[int, float] = field(default_factory=dict)

    def insert(self, vertex: int, key: float) -> None:
        """Insert or improve ``vertex`` with priority ``key``."""
        if key >= self.upper_bound:
            return

        current = self._best.get(vertex)
        if current is not None and key >= current:
            return

        self._best[vertex] = key
        heapq.heappush(self._heap, (key, vertex))

    def batch_prepend(self, items: list[tuple[int, float]]) -> None:
        """Reinsert a batch of vertices.

        The paper's data structure guarantees these items are all smaller than
        the remaining queue. For correctness, regular inserts are sufficient.
        """
        for vertex, key in items:
            self.insert(vertex, key)

    def _discard_stale(self) -> None:
        while self._heap:
            key, vertex = self._heap[0]
            if self._best.get(vertex) == key:
                return
            heapq.heappop(self._heap)

    def pull(self) -> tuple[list[int], float]:
        """Pull up to ``block_size`` minimum-key vertices.

        Returns:
            (vertices, bound) where:
            - ``vertices`` are up to the ``block_size`` smallest vertices.
            - ``bound`` is the separating upper bound for the *remaining*
              queue after removal, or ``upper_bound`` if the queue becomes
              empty.

        This matches the BMSSP recursion contract. The bound is *not* the
        current minimum key.
        """
        self._discard_stale()
        if not self._heap:
            return [], self.upper_bound

        group: list[int] = []
        while len(group) < self.block_size:
            self._discard_stale()
            if not self._heap:
                break

            key, vertex = heapq.heappop(self._heap)
            if self._best.get(vertex) != key:
                continue

            del self._best[vertex]
            group.append(vertex)

        self._discard_stale()
        next_bound = self._heap[0][0] if self._heap else self.upper_bound
        return group, next_bound

    def non_empty(self) -> bool:
        """Return True if at least one live entry remains."""
        self._discard_stale()
        return bool(self._heap)


# ---------------------------------------------------------------------------
# Algorithm sub-routines
# ---------------------------------------------------------------------------


def _base_case(
    graph: CSRGraph,
    bound: float,
    frontier: set[int],
    distances: list[float],
    predecessors: list[int],
    k: int,
) -> tuple[float, set[int]]:
    """Base case of BMSSP for a singleton frontier.

    This is *not* a full bounded Dijkstra. It settles at most ``k`` vertices
    whose shortest paths depend on the singleton source and returns the next
    separating boundary.

    Args:
        graph: The graph being traversed.
        bound: Upper distance limit; vertices at or beyond this distance
            are not explored.
        frontier: Set of source vertices for this bounded search.
        distances: Shared distance array, updated in place.
        predecessors: Shared predecessor array, updated in place.
        k: Unused. Kept for interface consistency with the recursive call.

    Returns:
        (bound_prime, completed)
        - If fewer than ``k`` vertices are found, returns ``(bound, completed)``.
        - Otherwise returns the next frontier boundary and the vertices with
          distance < ``bound_prime`` that were settled in this base case.
    """
    if len(frontier) != 1:
        raise ValueError("_base_case requires a singleton frontier")

    src = next(iter(frontier))
    heap: list[tuple[float, int]] = [(distances[src], src)]
    completed: set[int] = set()

    while heap and len(completed) < k:
        dist_u, u = heapq.heappop(heap)

        if dist_u != distances[u]:
            continue
        if dist_u >= bound:
            break

        for v, weight in graph.weighted_neighbors(u):
            candidate = dist_u + weight
            if candidate <= distances[v] and candidate < bound:
                distances[v] = candidate
                predecessors[v] = u
                heapq.heappush(heap, (candidate, v))

        # The singleton source is already complete by precondition; we only
        # count newly completed vertices toward the k-budget.
        if u != src:
            completed.add(u)

    if len(completed) < k:
        return bound, completed

    while heap and heap[0][0] != distances[heap[0][1]]:
        heapq.heappop(heap)

    bound_prime = heap[0][0] if heap else bound
    certified = {u for u in completed if distances[u] < bound_prime}
    return bound_prime, certified


def _find_pivots(
    graph: CSRGraph,
    bound: float,
    frontier: set[int],
    distances: list[float],
    predecessors: list[int],
    k: int,
) -> tuple[set[int], set[int]]:
    """Identify pivot vertices and the touched set via Bellman-Ford relaxation.

    We perform up to ``k`` Bellman-Ford-style waves from the current frontier
    and collect the completed set ``W``. For correctness, we conservatively
    return all frontier vertices as pivots.

    This is less selective than the paper's heavy-root pivot extraction, but
    it preserves correctness and is enough to make the Python port exact.

    Args:
        graph: The graph being traversed.
        bound: Distances at or above this value are ignored.
        frontier: Current set of active source vertices.
        distances: Shared distance array, updated in place.
        predecessors: Shared predecessor array, updated in place.
        k: Number of Bellman-Ford relaxation rounds.

    Returns:
        A ``(pivots, touched)`` pair. ``pivots`` is the full *frontier* —
        every source vertex is a candidate for recursive processing.
        ``touched`` is the set of vertices relaxed during the rounds whose
        distances were improved.
    """
    completed: set[int] = set(frontier)
    prev_wave: set[int] = set(frontier)
    frontier_size = max(1, len(frontier))

    for _ in range(k):
        current_wave: set[int] = set()

        for u in prev_wave:
            dist_u = distances[u]
            if dist_u >= bound:
                continue

            for v, weight in graph.weighted_neighbors(u):
                candidate = dist_u + weight
                if candidate <= distances[v] and candidate < bound:
                    distances[v] = candidate
                    predecessors[v] = u
                    if v not in completed:
                        completed.add(v)
                        current_wave.add(v)

        # Conservative early exit: same behavior as the Go reference in spirit.
        if len(completed) > k * frontier_size:
            return set(frontier), completed

        if not current_wave:
            break

        prev_wave = current_wave

    return set(frontier), completed


# ---------------------------------------------------------------------------
# Core recursive procedure
# ---------------------------------------------------------------------------


def _bmssp(
    graph: CSRGraph,
    level: int,
    bound: float,
    frontier: set[int],
    distances: list[float],
    predecessors: list[int],
    k: int,
    t: int,
) -> tuple[float, set[int]]:
    """
    Experimental deterministic SSSP implementation.

    This method is under active validation and is not currently part of the
    stable correctness contract of the library.

    Recursive BMSSP procedure (Algorithm 3 of the paper).

    Args:
        graph: The graph being traversed.
        level: Current recursion depth. Base case fires at ``level == 0``.
        bound: Upper distance bound for this call; vertices at or beyond
            this value are deferred to a higher level.
        frontier: Active source set for this recursive call.
        distances: Shared distance array, updated in place.
        predecessors: Shared predecessor array, updated in place.
        k: Frontier expansion factor (tuning constant).
        t: Recursion depth parameter (tuning constant).

    Returns:
        A ``(bound_prime, completed)`` pair where ``bound_prime`` is the
        tightest bound certified at this level and ``completed`` is the
        set of vertices settled.
    """
    if level == 0:
        return _base_case(graph, bound, frontier, distances, predecessors, k)

    pivots, touched = _find_pivots(graph, bound, frontier, distances, predecessors, k)

    queue = _LevelQueue(
        upper_bound=bound,
        block_size=2 ** max(0, (level - 1) * t),
    )
    for v in pivots:
        queue.insert(v, distances[v])

    completed: set[int] = set(touched)
    bound_prime = bound
    size_limit = k * k * (2 ** (level * t))

    while len(completed) < size_limit and queue.non_empty():
        sub_frontier_list, sub_bound = queue.pull()
        if not sub_frontier_list:
            break

        if level - 1 == 0:
            sub_bound_prime = sub_bound
            sub_completed: set[int] = set()

            for v in sub_frontier_list:
                bp, uc = _base_case(
                    graph,
                    sub_bound,
                    {v},
                    distances,
                    predecessors,
                    k,
                )
                if bp < sub_bound_prime:
                    sub_bound_prime = bp
                sub_completed |= uc
        else:
            sub_bound_prime, sub_completed = _bmssp(
                graph,
                level - 1,
                sub_bound,
                set(sub_frontier_list),
                distances,
                predecessors,
                k,
                t,
            )

        completed |= sub_completed

        candidates: list[tuple[int, float]] = []
        for u in sub_completed:
            dist_u = distances[u]
            for v, weight in graph.weighted_neighbors(u):
                candidate = dist_u + weight
                if candidate <= distances[v]:
                    distances[v] = candidate
                    predecessors[v] = u

                    if candidate < bound:
                        queue.insert(v, candidate)
                    elif sub_bound_prime <= candidate < sub_bound:
                        candidates.append((v, candidate))

        queue.batch_prepend(candidates)

        if sub_bound_prime < bound_prime:
            bound_prime = sub_bound_prime

        for x in touched:
            if distances[x] < bound_prime:
                completed.add(x)

    return bound_prime, completed
