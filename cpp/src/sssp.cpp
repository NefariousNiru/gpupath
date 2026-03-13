// file: cpp/src/sssp.cpp

#include "gpupath/sssp.hpp"

#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gpupath {
    /**
     * @brief Run Single-Source Shortest Path (SSSP) from @p source on a CSR graph.
     *
     * Validates the CSR structure and optional weight array before traversal,
     * then runs Dijkstra's algorithm using a binary min-heap. Neighbor indices
     * are bounds-checked during traversal to catch malformed graphs early.
     *
     * If @p weights is not provided, each edge is assigned a unit cost of `1.0`.
     *
     * @param num_vertices  Total number of vertices. Must be non-negative.
     * @param indptr        Row-pointer array of length `num_vertices + 1`.
     *                      Must be non-decreasing, start at `0`, and have
     *                      its last element equal to `indices.size()`.
     * @param indices       Flat neighbor array. Every entry must be in
     *                      `[0, num_vertices)`.
     * @param weights       Optional per-edge weights parallel to @p indices.
     *                      If provided, all values must be non-negative.
     * @param source        Source vertex. Must be in `[0, num_vertices)`.
     *
     * @return An `SsspResult` whose `distances[v]` holds the minimum path cost
     *         from @p source to vertex @p v, and whose `predecessors[v]` holds
     *         the vertex that last relaxed the edge into @p v. Unreachable
     *         vertices retain `+inf` in `distances` and `-1` in `predecessors`.
     *
     * @throws std::invalid_argument if @p num_vertices is negative, if
     *         @p indptr has the wrong size, is not non-decreasing, does not
     *         start at `0`, if its last element does not equal `indices.size()`,
     *         if @p weights has the wrong size, or if any weight is negative.
     * @throws std::out_of_range if @p source is outside `[0, num_vertices)`
     *         or if any neighbor index encountered during traversal is
     *         outside `[0, num_vertices)`.
     */
    SsspResult sssp(
        const int num_vertices,
        const std::vector<int> &indptr,
        const std::vector<int> &indices,
        const std::optional<std::vector<double> > &weights,
        int source
    ) {
        // --- CSR validation --------------------------------------------------

        if (num_vertices < 0) {
            throw std::invalid_argument("num_vertices must be non-negative");
        }

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

        for (int i = 0; i < num_vertices; ++i) {
            if (indptr[i] > indptr[i + 1]) {
                throw std::invalid_argument("indptr must be non-decreasing");
            }
        }

        if (weights.has_value()) {
            if (weights->size() != indices.size()) {
                throw std::invalid_argument("weights size must equal indices size");
            }

            for (std::size_t i = 0; i < weights->size(); ++i) {
                if ((*weights)[i] < 0.0) {
                    throw std::invalid_argument("weights must be non-negative");
                }
            }
        }

        // --- Dijkstra / SSSP -------------------------------------------------

        SsspResult result;
        result.distances.assign(
            num_vertices,
            std::numeric_limits<double>::infinity()
        );
        result.predecessors.assign(num_vertices, -1);

        using HeapEntry = std::pair<double, int>;
        std::priority_queue<
            HeapEntry,
            std::vector<HeapEntry>,
            std::greater<HeapEntry>
        > heap;

        result.distances[source] = 0.0;
        heap.emplace(0.0, source);

        while (!heap.empty()) {
            const auto [cur_dist, u] = heap.top();
            heap.pop();

            // Skip stale heap entries. This is the standard lazy-deletion
            // pattern used instead of a decrease-key operation.
            if (cur_dist > result.distances[u]) {
                continue;
            }

            for (int edge_idx = indptr[u]; edge_idx < indptr[u + 1]; ++edge_idx) {
                const int v = indices[edge_idx];

                // Bounds-check each neighbor to catch malformed index arrays.
                if (v < 0 || v >= num_vertices) {
                    throw std::out_of_range("neighbor index out of range");
                }

                const double weight = weights.has_value()
                                          ? (*weights)[edge_idx]
                                          : 1.0;

                // Strict improvement preserves deterministic predecessor updates
                // under a fixed adjacency iteration order.
                if (const double cand = cur_dist + weight; cand < result.distances[v]) {
                    result.distances[v] = cand;
                    result.predecessors[v] = u;
                    heap.emplace(cand, v);
                }
            }
        }

        return result;
    }
} // namespace gpupath
