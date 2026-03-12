# file: gpupath/types.py

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

    Attributes:
        num_vertices: Total number of vertices in the graph.
        indptr: Row-pointer array of length ``num_vertices + 1``.
        indices: Column-index (neighbour) array of length equal to the
            number of directed edges stored.
        directed: ``True`` if edges are one-directional; ``False`` if each
            edge is stored in both directions.
    """

    num_vertices: int
    indptr: list[int]
    indices: list[int]
    directed: bool = True

    @classmethod
    def from_csr(
        cls,
        *,
        indptr: Iterable[int],
        indices: Iterable[int],
        directed: bool = True,
    ) -> CSRGraph:
        """Construct a :class:`CSRGraph` directly from CSR arrays.

        Args:
            indptr: Row-pointer array. Must be non-decreasing, start with
                ``0``, and have its last element equal to ``len(indices)``.
            indices: Flat neighbour array. Every entry must be a valid
                vertex id in ``[0, num_vertices)``.
            directed: Whether the graph is directed. Defaults to ``True``.

        Returns:
            A validated :class:`CSRGraph` instance.

        Raises:
            ValueError: If ``indptr`` is empty, not non-decreasing, does not
                start with ``0``, or if ``indptr[-1] != len(indices)``.
            ValueError: If any entry in ``indices`` is outside the valid
                vertex-id range.
        """
        indptr_list = list(indptr)
        indices_list = list(indices)

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

        return cls(
            num_vertices=num_vertices,
            indptr=indptr_list,
            indices=indices_list,
            directed=directed,
        )

    @classmethod
    def from_edge_list(
        cls,
        *,
        num_vertices: int,
        edges: Iterable[tuple[int, int]],
        directed: bool = True,
    ) -> CSRGraph:
        """Construct a :class:`CSRGraph` from a sequence of ``(src, dst)`` edges.

        For undirected graphs (``directed=False``) each edge ``(u, v)`` is
        stored in both directions so that ``v`` appears in the neighbour list
        of ``u`` and vice versa.

        Args:
            num_vertices: Total number of vertices. Must be positive.
            edges: Iterable of ``(source, destination)`` integer pairs.
                Both endpoints must be valid vertex ids in
                ``[0, num_vertices)``.
            directed: Whether the graph is directed. Defaults to ``True``.

        Returns:
            A :class:`CSRGraph` built from the provided edge list.

        Raises:
            ValueError: If ``num_vertices <= 0`` or if any endpoint in
                ``edges`` is outside ``[0, num_vertices)``.
        """
        if num_vertices <= 0:
            raise ValueError("num_vertices must be positive")

        adjacency: list[list[int]] = [[] for _ in range(num_vertices)]

        for src, dst in edges:
            if src < 0 or src >= num_vertices:
                raise ValueError(f"source {src} out of range")
            if dst < 0 or dst >= num_vertices:
                raise ValueError(f"destination {dst} out of range")
            adjacency[src].append(dst)
            if not directed:
                adjacency[dst].append(src)

        indptr = [0]
        indices: list[int] = []
        for neighbors in adjacency:
            indices.extend(neighbors)
            indptr.append(len(indices))

        return cls(
            num_vertices=num_vertices,
            indptr=indptr,
            indices=indices,
            directed=directed,
        )

    def neighbors(self, vertex: int) -> list[int]:
        """Return the neighbours of *vertex* as a list of vertex ids.

        For directed graphs this returns only the *out-neighbours* of the
        given vertex (i.e. the destinations of edges that originate at
        ``vertex``).

        Args:
            vertex: The vertex whose neighbours are requested. Must be in
                ``[0, num_vertices)``.

        Returns:
            A list of integer vertex ids that are adjacent to ``vertex``.
            The list is a copy; mutating it does not affect the graph.

        Raises:
            IndexError: If ``vertex`` is outside ``[0, num_vertices)``.
        """
        if vertex < 0 or vertex >= self.num_vertices:
            raise IndexError(f"vertex {vertex} out of range")

        start = self.indptr[vertex]
        end = self.indptr[vertex + 1]
        return self.indices[start:end]
