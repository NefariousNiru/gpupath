#pragma once
// file: cpp/include/gpupath/bfs.hpp

#include <vector>

#include "gpupath/types.hpp"

namespace gpupath {
    /**
     * @brief Run Breadth-First Search from @p source on an unweighted CSR graph.
     *
     * Computes the shortest-hop distance and predecessor vertex for every
     * vertex reachable from @p source. Unreachable vertices retain `-1`
     * in both output arrays.
     *
     * Expected CSR invariants:
     * - `indptr.size() == num_vertices + 1`
     * - `indptr[0] == 0`
     * - `indptr` is non-decreasing
     * - `indptr.back() == indices.size()`
     * - every entry in @p indices is in `[0, num_vertices)`
     *
     * Complexity:
     * - Time: O(V + E)
     * - Auxiliary space: O(V)
     *
     * where V is @p num_vertices and E is `indices.size()`.
     *
     * @param num_vertices  Total number of vertices in the graph. Must be
     *                      non-negative.
     * @param indptr        CSR row-pointer array of length `num_vertices + 1`.
     *                      `indptr[v]` and `indptr[v + 1]` delimit the neighbors
     *                      of vertex @p v in @p indices.
     * @param indices       Flat CSR neighbor array. Every entry must be a valid
     *                      vertex id in `[0, num_vertices)`.
     * @param source        Source vertex from which BFS starts. Must be in
     *                      `[0, num_vertices)`.
     *
     * @return A `BfsResult` whose `distances[v]` holds the shortest-hop
     *         distance from @p source to vertex @p v, and whose
     *         `predecessors[v]` holds the vertex that first discovered @p v.
     *         Both arrays have length @p num_vertices.
     *
     * @throws std::invalid_argument if:
     *         - @p num_vertices is negative
     *         - @p indptr has incorrect size
     *         - @p indptr is empty
     *         - @p indptr does not start at `0`
     *         - @p indptr is not non-decreasing
     *         - @p indptr.back() does not equal `indices.size()`
     * @throws std::out_of_range if:
     *         - @p source is outside `[0, num_vertices)`
     *         - any neighbor index in @p indices is outside `[0, num_vertices)`
     */
    BfsResult bfs_unweighted(
        int num_vertices,
        const std::vector<int> &indptr,
        const std::vector<int> &indices,
        int source
    );
} // namespace gpupath
