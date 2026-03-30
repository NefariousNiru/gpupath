// file: cpp/include/gpupath/cuda_bfs.hpp

#pragma once

#include "gpupath/cuda_csr_graph.hpp"
#include "gpupath/types.hpp"

namespace gpupath {
    /**
     * @brief Run single-source unweighted BFS on a prepared CUDA CSR graph.
     *
     * This is the first real traversal entry point for the CUDA backend. The graph
     * is assumed to already reside on the device inside @ref CudaCsrGraph, so this
     * function performs traversal directly against prepared device CSR storage.
     *
     * Semantics:
     * - `distances[v]` is the minimum hop count from @p source to vertex @p v
     * - `predecessors[v]` stores the vertex that first discovered @p v
     * - unreachable vertices retain `-1` in both arrays
     * - the source retains predecessor `-1`
     *
     * Notes:
     * - This routine is for unweighted BFS only
     * - predecessor values may differ from the CPU serial BFS for ties, but they
     *   must still form valid shortest-path parent links
     *
     * @param graph Prepared CUDA CSR graph.
     * @param source Source vertex id in `[0, graph.num_vertices())`.
     *
     * @return A @ref BfsResult containing shortest-hop distances and predecessor
     *         links for all vertices in the graph.
     *
     * @throws std::out_of_range If @p source is outside the valid vertex range.
     * @throws std::invalid_argument If the graph is weighted.
     * @throws std::runtime_error On CUDA allocation/copy/kernel failures.
     */
    [[nodiscard]] BfsResult cuda_bfs_unweighted(const CudaCsrGraph &graph, int source);
} // namespace gpupath
