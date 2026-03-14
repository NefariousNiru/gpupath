// file: cpp/src/bfs.cpp

#include "gpupath/bfs.hpp"

#include <queue>
#include <stdexcept>

namespace gpupath {
    /**
     * @brief Run Breadth-First Search on a prepared CSR graph.
     *
     * Executes BFS directly on a prepared @ref NativeCsrGraph. Because the graph
     * has already been validated at construction time, this function can traverse
     * the native CSR storage directly without repeating Python-boundary CSR
     * validation work on every call.
     *
     * This function preserves the project BFS contract:
     * - `distances[v]` is the shortest-hop distance from @p source to @p v
     * - `predecessors[v]` is the predecessor on a valid shortest path
     * - unreachable vertices retain `-1` in both arrays
     * - the source retains predecessor `-1`
     *
     * @param graph Prepared native CSR graph.
     * @param source Source vertex in the range `[0, graph.num_vertices())`.
     *
     * @return A @ref BfsResult containing hop distances and predecessors for all
     *         vertices in the graph.
     *
     * @throws std::out_of_range If @p source is outside the valid vertex range.
     */
    BfsResult bfs_unweighted(const NativeCsrGraph &graph, const int source) {
        if (source < 0 || static_cast<std::size_t>(source) >= graph.num_vertices()) {
            throw std::out_of_range("source out of range");
        }

        BfsResult result;
        result.distances.assign(graph.num_vertices(), -1);
        result.predecessors.assign(graph.num_vertices(), -1);

        std::queue<int> q;
        result.distances[source] = 0;
        q.push(source);

        while (!q.empty()) {
            const int u = q.front();
            q.pop();

            for (int edge_idx = graph.indptr()[u]; edge_idx < graph.indptr()[u + 1]; ++edge_idx) {
                if (const int v = graph.indices()[edge_idx]; result.distances[v] == -1) {
                    result.distances[v] = result.distances[u] + 1;
                    result.predecessors[v] = u;
                    q.push(v);
                }
            }
        }

        return result;
    }
} // namespace gpupath
