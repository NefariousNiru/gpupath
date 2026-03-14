// file: cpp/src/bfs.cpp

#include "gpupath/bfs.hpp"

#include <queue>
#include <stdexcept>

namespace gpupath {
    /**
     * @brief Run Breadth-First Search from @p source on an unweighted CSR graph.
     *
     * Validates the CSR structure before traversal, then runs a standard
     * queue-based BFS. Neighbor indices are bounds-checked during traversal
     * to catch malformed graphs early.
     *
     * @param num_vertices  Total number of vertices. Must be non-negative.
     * @param indptr        Row-pointer array of length `num_vertices + 1`.
     *                      Must be non-decreasing, start at `0`, and have
     *                      its last element equal to `indices.size()`.
     * @param indices       Flat neighbor array. Every entry must be in
     *                      `[0, num_vertices)`.
     * @param source        Source vertex. Must be in `[0, num_vertices)`.
     *
     * @return A `BfsResult` whose `distances[v]` holds the shortest-hop
     *         distance from @p source to vertex @p v, and
     *         `predecessors[v]` holds the vertex that discovered @p v.
     *         Unreachable vertices retain `-1` in both arrays.
     *
     * @throws std::invalid_argument if @p num_vertices is negative, if
     *         @p indptr has the wrong size, is not non-decreasing, does not
     *         start at `0`, or if its last element does not equal
     *         `indices.size()`.
     * @throws std::out_of_range if @p source is outside `[0, num_vertices)`
     *         or if any neighbor index encountered during traversal is
     *         outside `[0, num_vertices)`.
     */
    BfsResult bfs_unweighted(
        const std::size_t num_vertices,
        const std::vector<int> &indptr,
        const std::vector<int> &indices,
        const int source
    ) {
        // --- CSR validation --------------------------------------------------

        if (source < 0 || source >= num_vertices) {
            throw std::out_of_range("source out of range");
        }

        if (static_cast<int>(indptr.size()) != num_vertices + 1) {
            throw std::invalid_argument("indptr size must equal num_vertices + 1");
        }

        if (indptr.empty()) {
            throw std::invalid_argument("indptr must not be empty");
        }

        if (indptr.front() != 0) {
            throw std::invalid_argument("indptr must start at 0");
        }

        if (indptr.back() != static_cast<int>(indices.size())) {
            throw std::invalid_argument("indptr last value must equal indices size");
        }

        for (std::size_t i = 0; i < num_vertices; ++i) {
            if (indptr[i] > indptr[i + 1]) {
                throw std::invalid_argument("indptr must be non-decreasing");
            }
        }

        // --- BFS -------------------------------------------------------------

        BfsResult result;
        result.distances.assign(num_vertices, -1);
        result.predecessors.assign(num_vertices, -1);

        std::queue<int> q;
        result.distances[source] = 0;
        q.push(source);

        while (!q.empty()) {
            const int u = q.front();
            q.pop();

            for (int edge_idx = indptr[u]; edge_idx < indptr[u + 1]; ++edge_idx) {
                const int v = indices[edge_idx];

                // Bounds-check each neighbor to catch malformed index arrays.
                if (v < 0 || v >= num_vertices) {
                    throw std::out_of_range("neighbor index out of range");
                }

                // First discovery fixes both shortest-hop distance and parent.
                if (result.distances[v] == -1) {
                    result.distances[v] = result.distances[u] + 1;
                    result.predecessors[v] = u;
                    q.push(v);
                }
            }
        }

        return result;
    }

    /**
     * @brief Run Breadth-First Search on a prepared CSR graph.
     *
     * Convenience overload that operates directly on a @ref NativeCsrGraph.
     * The graph is assumed to already satisfy all CSR invariants because
     * they are validated during construction of the NativeCsrGraph object.
     *
     * This simply forwards the graph's internal CSR storage to the
     * array-based BFS implementation.
     *
     * @param graph   Prepared CSR graph.
     * @param source  Source vertex in the range `[0, graph.num_vertices())`.
     *
     * @return A `BfsResult` whose `distances[v]` contains the shortest-hop
     *         distance from @p source to vertex @p v, and whose
     *         `predecessors[v]` contains the vertex that discovered @p v.
     *         Unreachable vertices retain `-1` in both arrays.
     *
     * @throws std::out_of_range if @p source is outside the valid range.
     */
    BfsResult bfs_unweighted(const NativeCsrGraph &graph, const int source) {
        return bfs_unweighted(
            graph.num_vertices(),
            graph.indptr(),
            graph.indices(),
            source
        );
    }
} // namespace gpupath
