// file: cpp/include/gpupath/sssp.hpp

#pragma once

#include "gpupath/native_csr_graph.hpp"
#include "gpupath/types.hpp"

namespace gpupath {
    /**
     * @brief Run single-source shortest path from a prepared native CSR graph.
     *
     * This is the native single-source weighted traversal entry point used by the
     * Python native engine. The graph is assumed to have already passed CSR and
     * weight validation during construction of the @ref NativeCsrGraph object, so
     * this function operates directly on native-resident CSR storage without
     * rebuilding or revalidating graph structure on every call.
     *
     * If the prepared graph stores explicit weights, those weights are used. If the
     * graph is unweighted, each edge is treated as having implicit cost `1.0`.
     *
     * Semantics:
     * - `distances[v]` is the minimum path cost from @p source to vertex @p v
     * - `predecessors[v]` stores the vertex that last relaxed the edge into @p v
     * - unreachable vertices retain `+inf` in `distances` and `-1` in
     *   `predecessors`
     * - the source vertex retains predecessor `-1`
     *
     * Complexity:
     * - Time: O((V + E) log V)
     * - Auxiliary space: O(V)
     *
     * where V is `graph.num_vertices()` and E is `graph.num_edges()`.
     *
     * @param graph Prepared native CSR graph.
     * @param source Source vertex id in `[0, graph.num_vertices())`.
     *
     * @return An @ref SsspResult containing shortest-path distances and predecessor
     *         links for all vertices in the graph.
     *
     * @throws std::out_of_range If @p source is outside
     *         `[0, graph.num_vertices())`.
     */
    [[nodiscard]] SsspResult sssp(const NativeCsrGraph &graph, int source);
} // namespace gpupath
