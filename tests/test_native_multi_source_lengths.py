# file: tests/test_native_multi_source_lengths.py

from __future__ import annotations

import math

import pytest

from gpupath.engine.native import NativePathEngine
from gpupath.engine.reference import ReferencePathEngine
from gpupath.graph import CSRGraph
from gpupath.query import _cost_matrix

# ============================================================================
# Helpers
# ============================================================================


def _assert_matrix_equal(
    actual: list[list[int]] | list[list[float]],
    expected: list[list[int]] | list[list[float]],
) -> None:
    assert len(actual) == len(expected), (actual, expected)

    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert len(actual_row) == len(expected_row), (actual_row, expected_row)

        for actual_value, expected_value in zip(actual_row, expected_row, strict=True):
            if isinstance(expected_value, float):
                if math.isinf(expected_value):
                    assert math.isinf(actual_value), (actual_value, expected_value)
                else:
                    assert actual_value == pytest.approx(expected_value), (
                        actual_value,
                        expected_value,
                    )
            else:
                assert actual_value == expected_value, (actual_value, expected_value)


def _build_unweighted_directed_graph() -> CSRGraph:
    """
    Directed graph with 6 vertices.

    Edges:
        0 -> 1, 2
        1 -> 3
        2 -> 3
        3 -> 4
        4 -> (none)
        5 -> (isolated)
    """
    return CSRGraph(
        num_vertices=6,
        indptr=[0, 2, 3, 4, 5, 5, 5],
        indices=[1, 2, 3, 3, 4],
        directed=True,
    )


def _build_weighted_directed_graph() -> CSRGraph:
    """
    Directed weighted graph with 6 vertices.

    Edges:
        0 -> 1 (1.0), 2 (5.0)
        1 -> 2 (1.5), 3 (2.0)
        2 -> 3 (1.0)
        3 -> 4 (3.5)
        4 -> (none)
        5 -> (isolated)
    """
    return CSRGraph(
        num_vertices=6,
        indptr=[0, 2, 4, 5, 6, 6, 6],
        indices=[1, 2, 2, 3, 3, 4],
        weights=[1.0, 5.0, 1.5, 2.0, 1.0, 3.5],
        directed=True,
    )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def reference_engine() -> ReferencePathEngine:
    return ReferencePathEngine()


@pytest.fixture
def native_engine() -> NativePathEngine:
    return NativePathEngine()


@pytest.fixture
def unweighted_graph() -> CSRGraph:
    return _build_unweighted_directed_graph()


@pytest.fixture
def weighted_graph() -> CSRGraph:
    return _build_weighted_directed_graph()


# ============================================================================
# Engine-level parity: unweighted
# ============================================================================


def test_multi_source_lengths_unweighted_targets_none_parity(
    reference_engine: ReferencePathEngine,
    native_engine: NativePathEngine,
    unweighted_graph: CSRGraph,
) -> None:
    sources = [0, 2, 5]

    expected = reference_engine.multi_source_lengths(
        unweighted_graph,
        sources,
        targets=None,
        method="default",
    )
    actual = native_engine.multi_source_lengths(
        unweighted_graph,
        sources,
        targets=None,
        method="default",
    )

    _assert_matrix_equal(actual, expected)

    # Spot-check semantics
    assert actual[0] == [0, 1, 1, 2, 3, -1]
    assert actual[1] == [-1, -1, 0, 1, 2, -1]
    assert actual[2] == [-1, -1, -1, -1, -1, 0]


def test_multi_source_lengths_unweighted_target_subset_order_and_duplicates(
    reference_engine: ReferencePathEngine,
    native_engine: NativePathEngine,
    unweighted_graph: CSRGraph,
) -> None:
    sources = [0, 2, 0]
    targets = [4, 1, 4, 5]

    expected = reference_engine.multi_source_lengths(
        unweighted_graph,
        sources,
        targets=targets,
        method="default",
    )
    actual = native_engine.multi_source_lengths(
        unweighted_graph,
        sources,
        targets=targets,
        method="default",
    )

    _assert_matrix_equal(actual, expected)

    assert actual == [
        [3, 1, 3, -1],
        [2, -1, 2, -1],
        [3, 1, 3, -1],
    ]


def test_multi_source_lengths_unweighted_empty_sources(
    reference_engine: ReferencePathEngine,
    native_engine: NativePathEngine,
    unweighted_graph: CSRGraph,
) -> None:
    expected = reference_engine.multi_source_lengths(
        unweighted_graph,
        [],
        targets=[0, 1],
        method="default",
    )
    actual = native_engine.multi_source_lengths(
        unweighted_graph,
        [],
        targets=[0, 1],
        method="default",
    )

    assert expected == []
    assert actual == []


def test_multi_source_lengths_unweighted_empty_targets(
    reference_engine: ReferencePathEngine,
    native_engine: NativePathEngine,
    unweighted_graph: CSRGraph,
) -> None:
    sources = [0, 2, 5]
    targets: list[int] = []

    expected = reference_engine.multi_source_lengths(
        unweighted_graph,
        sources,
        targets=targets,
        method="default",
    )
    actual = native_engine.multi_source_lengths(
        unweighted_graph,
        sources,
        targets=targets,
        method="default",
    )

    assert expected == [[], [], []]
    assert actual == [[], [], []]


# ============================================================================
# Engine-level parity: weighted
# ============================================================================


def test_multi_source_lengths_weighted_targets_none_parity(
    reference_engine: ReferencePathEngine,
    native_engine: NativePathEngine,
    weighted_graph: CSRGraph,
) -> None:
    sources = [0, 2, 5]

    expected = reference_engine.multi_source_lengths(
        weighted_graph,
        sources,
        targets=None,
        method="default",
    )
    actual = native_engine.multi_source_lengths(
        weighted_graph,
        sources,
        targets=None,
        method="default",
    )

    _assert_matrix_equal(actual, expected)

    # Spot-check semantics
    assert actual[0][0] == pytest.approx(0.0)
    assert actual[0][1] == pytest.approx(1.0)
    assert actual[0][2] == pytest.approx(2.5)
    assert actual[0][3] == pytest.approx(3.0)
    assert actual[0][4] == pytest.approx(6.5)
    assert math.isinf(actual[0][5])

    assert math.isinf(actual[1][0])
    assert math.isinf(actual[1][1])
    assert actual[1][2] == pytest.approx(0.0)
    assert actual[1][3] == pytest.approx(1.0)
    assert actual[1][4] == pytest.approx(4.5)
    assert math.isinf(actual[1][5])

    assert math.isinf(actual[2][0])
    assert math.isinf(actual[2][1])
    assert math.isinf(actual[2][2])
    assert math.isinf(actual[2][3])
    assert math.isinf(actual[2][4])
    assert actual[2][5] == pytest.approx(0.0)


def test_multi_source_lengths_weighted_target_subset_order_and_duplicates(
    reference_engine: ReferencePathEngine,
    native_engine: NativePathEngine,
    weighted_graph: CSRGraph,
) -> None:
    sources = [0, 1, 0]
    targets = [4, 2, 4, 5]

    expected = reference_engine.multi_source_lengths(
        weighted_graph,
        sources,
        targets=targets,
        method="default",
    )
    actual = native_engine.multi_source_lengths(
        weighted_graph,
        sources,
        targets=targets,
        method="default",
    )

    _assert_matrix_equal(actual, expected)

    assert actual[0][0] == pytest.approx(6.5)
    assert actual[0][1] == pytest.approx(2.5)
    assert actual[0][2] == pytest.approx(6.5)
    assert math.isinf(actual[0][3])

    assert actual[1][0] == pytest.approx(5.5)
    assert actual[1][1] == pytest.approx(1.5)
    assert actual[1][2] == pytest.approx(5.5)
    assert math.isinf(actual[1][3])

    assert actual[2][0] == pytest.approx(6.5)
    assert actual[2][1] == pytest.approx(2.5)
    assert actual[2][2] == pytest.approx(6.5)
    assert math.isinf(actual[2][3])


def test_multi_source_lengths_weighted_empty_sources(
    reference_engine: ReferencePathEngine,
    native_engine: NativePathEngine,
    weighted_graph: CSRGraph,
) -> None:
    expected = reference_engine.multi_source_lengths(
        weighted_graph,
        [],
        targets=[0, 1],
        method="default",
    )
    actual = native_engine.multi_source_lengths(
        weighted_graph,
        [],
        targets=[0, 1],
        method="default",
    )

    assert expected == []
    assert actual == []


def test_multi_source_lengths_weighted_empty_targets(
    reference_engine: ReferencePathEngine,
    native_engine: NativePathEngine,
    weighted_graph: CSRGraph,
) -> None:
    sources = [0, 2, 5]
    targets: list[int] = []

    expected = reference_engine.multi_source_lengths(
        weighted_graph,
        sources,
        targets=targets,
        method="default",
    )
    actual = native_engine.multi_source_lengths(
        weighted_graph,
        sources,
        targets=targets,
        method="default",
    )

    assert expected == [[], [], []]
    assert actual == [[], [], []]


# ============================================================================
# Public API parity
# ============================================================================


def test_public_cost_matrix_unweighted_matches_reference_and_native(
    reference_engine: ReferencePathEngine,
    native_engine: NativePathEngine,
    unweighted_graph: CSRGraph,
) -> None:
    sources = [0, 2, 5]
    targets = [4, 1, 5]

    reference_direct = reference_engine.multi_source_lengths(
        unweighted_graph,
        sources,
        targets=targets,
        method="default",
    )
    native_direct = native_engine.multi_source_lengths(
        unweighted_graph,
        sources,
        targets=targets,
        method="default",
    )
    public_reference = _cost_matrix(
        unweighted_graph,
        sources=sources,
        targets=targets,
        engine=reference_engine,
        method="default",
    )
    public_native = _cost_matrix(
        unweighted_graph,
        sources=sources,
        targets=targets,
        engine=native_engine,
        method="default",
    )

    _assert_matrix_equal(native_direct, reference_direct)
    _assert_matrix_equal(public_reference, reference_direct)
    _assert_matrix_equal(public_native, reference_direct)


def test_public_cost_matrix_weighted_matches_reference_and_native(
    reference_engine: ReferencePathEngine,
    native_engine: NativePathEngine,
    weighted_graph: CSRGraph,
) -> None:
    sources = [0, 1, 5]
    targets = [4, 2, 5]

    reference_direct = reference_engine.multi_source_lengths(
        weighted_graph,
        sources,
        targets=targets,
        method="default",
    )
    native_direct = native_engine.multi_source_lengths(
        weighted_graph,
        sources,
        targets=targets,
        method="default",
    )
    public_reference = _cost_matrix(
        weighted_graph,
        sources=sources,
        targets=targets,
        engine=reference_engine,
        method="default",
    )
    public_native = _cost_matrix(
        weighted_graph,
        sources=sources,
        targets=targets,
        engine=native_engine,
        method="default",
    )

    _assert_matrix_equal(native_direct, reference_direct)
    _assert_matrix_equal(public_reference, reference_direct)
    _assert_matrix_equal(public_native, reference_direct)


# ============================================================================
# Validation / error behavior parity
# ============================================================================


@pytest.mark.parametrize("bad_sources", [[-1], [6], [0, 6]])
def test_multi_source_lengths_invalid_sources_raise_for_both_engines(
    reference_engine: ReferencePathEngine,
    native_engine: NativePathEngine,
    unweighted_graph: CSRGraph,
    bad_sources: list[int],
) -> None:
    with pytest.raises((ValueError, IndexError)):
        reference_engine.multi_source_lengths(
            unweighted_graph,
            bad_sources,
            targets=[0, 1],
            method="default",
        )

    with pytest.raises((ValueError, IndexError)):
        native_engine.multi_source_lengths(
            unweighted_graph,
            bad_sources,
            targets=[0, 1],
            method="default",
        )


@pytest.mark.parametrize("bad_targets", [[-1], [6], [1, 6]])
def test_multi_source_lengths_invalid_targets_raise_for_both_engines(
    reference_engine: ReferencePathEngine,
    native_engine: NativePathEngine,
    weighted_graph: CSRGraph,
    bad_targets: list[int],
) -> None:
    with pytest.raises((ValueError, IndexError)):
        reference_engine.multi_source_lengths(
            weighted_graph,
            [0, 1],
            targets=bad_targets,
            method="default",
        )

    with pytest.raises((ValueError, IndexError)):
        native_engine.multi_source_lengths(
            weighted_graph,
            [0, 1],
            targets=bad_targets,
            method="default",
        )


def test_native_multi_source_lengths_rejects_bmssp_for_now(
    native_engine: NativePathEngine,
    weighted_graph: CSRGraph,
) -> None:
    with pytest.raises(NotImplementedError):
        native_engine.multi_source_lengths(
            weighted_graph,
            [0, 1],
            targets=[2, 4],
            method="bmssp",
        )


# ============================================================================
# Reuse / system-level sanity
# ============================================================================


def test_native_multi_source_lengths_is_stable_across_repeated_calls(
    native_engine: NativePathEngine,
    weighted_graph: CSRGraph,
) -> None:
    sources = [0, 1, 2, 5]
    targets = [4, 3, 2, 1, 0]

    first = native_engine.multi_source_lengths(
        weighted_graph,
        sources,
        targets=targets,
        method="default",
    )
    second = native_engine.multi_source_lengths(
        weighted_graph,
        sources,
        targets=targets,
        method="default",
    )
    third = native_engine.multi_source_lengths(
        weighted_graph,
        sources,
        targets=None,
        method="default",
    )

    _assert_matrix_equal(first, second)
    assert len(third) == len(sources)
    assert all(len(row) == weighted_graph.num_vertices for row in third)
