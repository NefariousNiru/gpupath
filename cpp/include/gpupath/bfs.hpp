// file: cpp/include/gpupath/bfs.hpp

#pragma once

#include "gpupath/native_csr_graph.hpp"
#include "gpupath/types.hpp"

namespace gpupath {
    /**
     * @brief Run BFS from a source on a prepared native CSR graph.
     *
     * This is the native single-source unweighted traversal entry point used by
     * the Python native engine. The graph is assumed to have already passed CSR
     * validation during construction of the @ref NativeCsrGraph object, so this
     * function operates directly on native-resident CSR storage without rebuilding
     * or revalidating the graph structure from Python inputs.
     *
     * Semantics:
     * - `distances[v]` is the minimum hop count from @p source to vertex @p v
     * - `predecessors[v]` stores the vertex that first discovered @p v
     * - unreachable vertices retain `-1` in both arrays
     * - the source vertex retains predecessor `-1`
     *
     * Complexity:
     * - Time: O(V + E)
     * - Auxiliary space: O(V)
     *
     * where V is `graph.num_vertices()` and E is `graph.num_edges()`.
     *
     * @param graph Prepared native CSR graph.
     * @param source Source vertex id in `[0, graph.num_vertices())`.
     *
     * @return A @ref BfsResult containing shortest-hop distances and predecessor
     *         links for all vertices in the graph.
     *
     * @throws std::out_of_range If @p source is outside
     *         `[0, graph.num_vertices())`.
     */
    [[nodiscard]] BfsResult bfs_unweighted(const NativeCsrGraph &graph, int source);
} // namespace gpupath
