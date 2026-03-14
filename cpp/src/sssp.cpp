// file: cpp/src/sssp.cpp

#include "gpupath/native_csr_graph.hpp"
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
     * @brief Run single-source shortest path on a prepared CSR graph.
     *
     * Executes Dijkstra-style single-source shortest path directly on a prepared
     * @ref NativeCsrGraph. Because the graph has already been validated at
     * construction time, this function can traverse native CSR storage directly
     * without repeating Python-boundary CSR or weight validation work on every
     * call.
     *
     * If the graph stores explicit edge weights, those weights are used. Otherwise,
     * every edge is treated as having implicit unit cost `1.0`.
     *
     * This function preserves the project SSSP contract:
     * - `distances[v]` is the minimum path cost from @p source to @p v
     * - `predecessors[v]` is the predecessor on a valid shortest path
     * - unreachable vertices retain `+inf` in `distances` and `-1` in
     *   `predecessors`
     * - the source retains predecessor `-1`
     *
     * @param graph Prepared native CSR graph.
     * @param source Source vertex in the range `[0, graph.num_vertices())`.
     *
     * @return An @ref SsspResult containing shortest-path distances and
     *         predecessors for all vertices in the graph.
     *
     * @throws std::out_of_range If @p source is outside the valid vertex range.
     */
    SsspResult sssp(const NativeCsrGraph &graph, const int source) {
        if (source < 0 || static_cast<std::size_t>(source) >= graph.num_vertices()) {
            throw std::out_of_range("source out of range");
        }

        SsspResult result;
        result.distances.assign(
            graph.num_vertices(),
            std::numeric_limits<double>::infinity()
        );
        result.predecessors.assign(graph.num_vertices(), -1);

        using HeapEntry = std::pair<double, int>;
        std::priority_queue<
            HeapEntry,
            std::vector<HeapEntry>,
            std::greater<>
        > heap;

        result.distances[source] = 0.0;
        heap.emplace(0.0, source);

        const bool has_weights = graph.weights().has_value();

        while (!heap.empty()) {
            const auto [cur_dist, u] = heap.top();
            heap.pop();

            if (cur_dist > result.distances[u]) {
                continue;
            }

            for (int edge_idx = graph.indptr()[u]; edge_idx < graph.indptr()[u + 1]; ++edge_idx) {
                const int v = graph.indices()[edge_idx];
                const double weight = has_weights ? (*graph.weights())[edge_idx] : 1.0;

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
