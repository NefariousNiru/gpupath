# file: gpupath/types.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, NewType

VertexId = NewType("VertexId", int)
"""A type alias representing a vertex identifier in a graph."""

UNREACHABLE_DISTANCE: int = -1
"""Sentinel value indicating that a vertex is unreachable from the BFS source."""

NO_PREDECESSOR: int = -1
"""Sentinel value indicating that a vertex has no predecessor (i.e. it is the source or unreachable)."""

INF_FLOAT = float("inf")
"""Float +ve infinity"""


@dataclass(frozen=True, slots=True)
class BfsResult:
    """Result of a Breadth-First Search (BFS) traversal on a graph.

    Attributes:
        distances: A list where ``distances[v]`` holds the shortest distance
            (in number of edges) from the BFS source vertex to vertex ``v``.
            A value of ``UNREACHABLE_DISTANCE`` indicates vertex ``v`` was
            not reachable from the source.
        predecessors: A list where ``predecessors[v]`` holds the predecessor
            of vertex ``v`` along the shortest path from the BFS source.
            A value of ``NO_PREDECESSOR`` indicates that ``v`` is either
            the source vertex or was not reachable.
    """

    distances: List[int]
    predecessors: List[int]


@dataclass(frozen=True, slots=True)
class SsspResult:
    """Stores the result of a Single-Source Shortest Path traversal.

    Attributes:
        distances: A list where ``distances[v]`` holds the minimum path cost
            from the source to vertex ``v``. Unreachable vertices have value
            ``INF_FLOAT``.
        predecessors: A list where ``predecessors[v]`` holds the predecessor
            of vertex ``v`` on one shortest path from the source. A value of
            ``NO_PREDECESSOR`` indicates that ``v`` is the source or is
            unreachable.
    """

    distances: List[float]
    predecessors: List[int]
