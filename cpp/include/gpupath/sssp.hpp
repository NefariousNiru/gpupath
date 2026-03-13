#pragma once
// file: cpp/include/gpupath/sssp.hpp

#include <optional>
#include <vector>

#include "gpupath/types.hpp"

namespace gpupath {
    /**
     * @brief Run Single-Source Shortest Path (SSSP) from @p source on a CSR graph.
     *
     * Computes the minimum-cost distance and predecessor vertex for every vertex
     * reachable from @p source. The current implementation uses Dijkstra's
     * algorithm with a binary min-heap and therefore requires all edge weights
     * to be non-negative.
     *
     * Weight semantics:
     * - If @p weights is provided, it must be parallel to @p indices and each
     *   entry is used as the edge cost.
     * - If @p weights is not provided, the graph is treated as unweighted and
     *   every edge is assigned cost `1.0`.
     *
     * Expected CSR invariants:
     * - `indptr.size() == num_vertices + 1`
     * - `indptr[0] == 0`
     * - `indptr` is non-decreasing
     * - `indptr.back() == indices.size()`
     * - every entry in @p indices is in `[0, num_vertices)`
     * - if @p weights is provided, `weights->size() == indices.size()`
     * - if @p weights is provided, every weight is non-negative
     *
     * Complexity:
     * - Time: O((V + E) log V)
     * - Auxiliary space: O(V)
     *
     * where V is @p num_vertices and E is `indices.size()`.
     *
     * @param num_vertices  Total number of vertices in the graph. Must be
     *                      non-negative.
     * @param indptr        CSR row-pointer array of length `num_vertices + 1`.
     *                      `indptr[v]` and `indptr[v + 1]` delimit the outgoing
     *                      edges of vertex @p v in @p indices.
     * @param indices       Flat CSR neighbor array. Every entry must be a valid
     *                      vertex id in `[0, num_vertices)`.
     * @param weights       Optional flat edge-weight array parallel to @p indices.
     *                      If omitted, every edge is treated as weight `1.0`.
     * @param source        Source vertex from which SSSP starts. Must be in
     *                      `[0, num_vertices)`.
     *
     * @return An `SsspResult` whose `distances[v]` holds the minimum path cost
     *         from @p source to vertex @p v, and whose `predecessors[v]` holds
     *         the vertex that last relaxed the edge into @p v. Both arrays have
     *         length @p num_vertices.
     *
     * @throws std::invalid_argument if:
     *         - @p num_vertices is negative
     *         - @p indptr has incorrect size
     *         - @p indptr is empty
     *         - @p indptr does not start at `0`
     *         - @p indptr is not non-decreasing
     *         - @p indptr.back() does not equal `indices.size()`
     *         - @p weights is provided but its size differs from `indices.size()`
     *         - any provided weight is negative
     * @throws std::out_of_range if:
     *         - @p source is outside `[0, num_vertices)`
     *         - any neighbor index in @p indices is outside `[0, num_vertices)`
     */
    SsspResult sssp(
        int num_vertices,
        const std::vector<int> &indptr,
        const std::vector<int> &indices,
        const std::optional<std::vector<double> > &weights,
        int source
    );
} // namespace gpupath
