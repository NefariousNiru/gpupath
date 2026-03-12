# file: gpupath/graph.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class CSRGraph:
    """A graph stored in Compressed Sparse Row (CSR) format.

    CSR is a memory-efficient representation for sparse graphs. Adjacency
    information is encoded in two parallel arrays:

    - ``indptr``: a length ``(num_vertices + 1)`` array of offsets into
      ``indices``. The neighbours of vertex ``v`` are stored at
      ``indices[indptr[v] : indptr[v + 1]]``.
    - ``indices``: a flat array of destination vertex ids.

    Optionally, a ``weights`` array of the same length as ``indices`` stores
    the cost of each edge. If ``weights`` is ``None`` the graph is unweighted
    and every edge is treated as having cost ``1.0``.

    Attributes:
        num_vertices: Total number of vertices in the graph.
        indptr: Row-pointer array of length ``num_vertices + 1``.
        indices: Column-index (neighbour) array of length equal to the
            number of directed edges stored.
        weights: Per-edge cost array, parallel to ``indices``. ``None`` for
            unweighted graphs. All values must be non-negative.
        directed: ``True`` if edges are one-directional; ``False`` if each
            edge is stored in both directions.
    """

    num_vertices: int
    indptr: list[int]
    indices: list[int]
    weights: list[float] | None = None
    directed: bool = True

    @classmethod
    def from_csr(
        cls,
        *,
        indptr: Iterable[int],
        indices: Iterable[int],
        weights: Iterable[float] | None = None,
        directed: bool = True,
    ) -> CSRGraph:
        """Construct a :class:`CSRGraph` directly from CSR arrays.

        Args:
            indptr: Row-pointer array. Must be non-decreasing, start with
                ``0``, and have its last element equal to ``len(indices)``.
            indices: Flat neighbour array. Every entry must be a valid
                vertex id in ``[0, num_vertices)``.
            weights: Optional per-edge cost array, parallel to ``indices``.
                All values must be non-negative. Pass ``None`` for an
                unweighted graph.
            directed: Whether the graph is directed. Defaults to ``True``.

        Returns:
            A validated :class:`CSRGraph` instance.

        Raises:
            ValueError: If ``indptr`` is empty, not non-decreasing, does not
                start with ``0``, or if ``indptr[-1] != len(indices)``.
            ValueError: If any entry in ``indices`` is outside
                ``[0, num_vertices)``.
            ValueError: If ``weights`` is provided but its length differs
                from ``len(indices)``, or if any weight is negative.
        """
        indptr_list = list(indptr)
        indices_list = list(indices)
        weights_list = list(weights) if weights is not None else None

        if not indptr_list:
            raise ValueError("indptr must not be empty")

        num_vertices = len(indptr_list) - 1

        if num_vertices < 0:
            raise ValueError("indptr must contain at least one element")
        if indptr_list[0] != 0:
            raise ValueError("indptr[0] must be 0")
        if indptr_list[-1] != len(indices_list):
            raise ValueError("indptr[-1] must equal len(indices)")

        prev = 0
        for i, value in enumerate(indptr_list):
            if value < prev:
                raise ValueError(
                    f"indptr must be non-decreasing; bad value at index {i}"
                )
            prev = value

        for i, dst in enumerate(indices_list):
            if dst < 0 or dst >= num_vertices:
                raise ValueError(
                    f"indices[{i}]={dst} out of range for {num_vertices} vertices"
                )

        if weights_list is not None:
            if len(weights_list) != len(indices_list):
                raise ValueError("weights must have the same length as indices")
            for i, w in enumerate(weights_list):
                if w < 0:
                    raise ValueError(
                        f"weights[{i}]={w} is negative; only non-negative weights are supported"
                    )

        return cls(
            num_vertices=num_vertices,
            indptr=indptr_list,
            indices=indices_list,
            weights=weights_list,
            directed=directed,
        )

    @classmethod
    def from_edge_list(
        cls,
        *,
        num_vertices: int,
        edges: Iterable[tuple[int, int] | tuple[int, int, float]],
        directed: bool = True,
    ) -> CSRGraph:
        """Construct a :class:`CSRGraph` from a sequence of edges.

        Each edge may be supplied as a ``(src, dst)`` pair for an unweighted
        graph, or as a ``(src, dst, weight)`` triple for a weighted one.
        Mixing weighted and unweighted edges in the same call is permitted;
        unweighted edges default to cost ``1.0``.

        For undirected graphs (``directed=False``) each edge is stored in
        both directions so that ``dst`` appears in the neighbour list of
        ``src`` and vice versa.

        Args:
            num_vertices: Total number of vertices. Must be positive.
            edges: Iterable of ``(src, dst)`` or ``(src, dst, weight)``
                tuples. Both endpoints must be valid vertex ids in
                ``[0, num_vertices)``. All weights must be non-negative.
            directed: Whether the graph is directed. Defaults to ``True``.

        Returns:
            A :class:`CSRGraph` built from the provided edge list. The
            ``weights`` field is ``None`` when no edge supplies a weight.

        Raises:
            ValueError: If ``num_vertices <= 0``.
            ValueError: If any endpoint is outside ``[0, num_vertices)``.
            ValueError: If any weight is negative.
            ValueError: If an edge tuple has a length other than 2 or 3.
        """
        if num_vertices <= 0:
            raise ValueError("num_vertices must be positive")

        adjacency: list[list[tuple[int, float | None]]] = [
            [] for _ in range(num_vertices)
        ]
        saw_weight = False

        for edge in edges:
            if len(edge) == 2:
                src, dst = edge
                weight = None
            elif len(edge) == 3:
                src, dst, weight = edge
                saw_weight = True
                if weight < 0:
                    raise ValueError("only non-negative weights are supported")
            else:
                raise ValueError(
                    "each edge must be a (src, dst) or (src, dst, weight) tuple"
                )

            if src < 0 or src >= num_vertices:
                raise ValueError(f"source {src} out of range")
            if dst < 0 or dst >= num_vertices:
                raise ValueError(f"destination {dst} out of range")

            adjacency[src].append((dst, weight))
            if not directed:
                adjacency[dst].append((src, weight))

        indptr = [0]
        indices: list[int] = []
        weights: list[float] | None = [] if saw_weight else None

        for neighbors in adjacency:
            for dst, weight in neighbors:
                indices.append(dst)
                if weights is not None:
                    weights.append(1.0 if weight is None else float(weight))
            indptr.append(len(indices))

        return cls(
            num_vertices=num_vertices,
            indptr=indptr,
            indices=indices,
            weights=weights,
            directed=directed,
        )

    def neighbors(self, vertex: int) -> list[int]:
        """Return the neighbours of *vertex* as a list of vertex ids.

        For directed graphs this returns only the out-neighbours of
        *vertex*, i.e. the destinations of edges that originate at
        *vertex*.

        Args:
            vertex: The vertex whose neighbours are requested. Must be in
                ``[0, num_vertices)``.

        Returns:
            A list of integer vertex ids adjacent to *vertex*. The list is
            a copy; mutating it does not affect the graph.

        Raises:
            IndexError: If *vertex* is outside ``[0, num_vertices)``.
        """
        if vertex < 0 or vertex >= self.num_vertices:
            raise IndexError(f"vertex {vertex} out of range")

        start = self.indptr[vertex]
        end = self.indptr[vertex + 1]
        return self.indices[start:end]

    def weighted_neighbors(self, vertex: int) -> list[tuple[int, float]]:
        """Return the neighbours of *vertex* together with their edge weights.

        Behaves like :meth:`neighbors` but returns ``(destination, weight)``
        pairs. For unweighted graphs every edge is reported with weight
        ``1.0``.

        Args:
            vertex: The vertex whose weighted neighbours are requested.
                Must be in ``[0, num_vertices)``.

        Returns:
            A list of ``(destination, weight)`` tuples for every edge
            leaving *vertex*. The list is a copy; mutating it does not
            affect the graph.

        Raises:
            IndexError: If *vertex* is outside ``[0, num_vertices)``.
        """
        if vertex < 0 or vertex >= self.num_vertices:
            raise IndexError(f"vertex {vertex} out of range")

        start = self.indptr[vertex]
        end = self.indptr[vertex + 1]

        if self.weights is None:
            return [(dst, 1.0) for dst in self.indices[start:end]]

        return list(zip(self.indices[start:end], self.weights[start:end], strict=False))

    @property
    def is_weighted(self) -> bool:
        """``True`` if this graph carries per-edge weights, ``False`` otherwise."""
        return self.weights is not None
