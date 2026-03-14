// file: multi_source_lengths.hpp
#pragma once

#include <optional>
#include <vector>

#include "gpupath/native_csr_graph.hpp"

namespace gpupath {
    /**
     * @brief Compute unweighted shortest-path lengths for multiple sources.
     *
     * Executes one BFS per source and assembles a dense distance matrix whose row
     * order matches @p sources exactly. If @p targets is provided, each row is
     * filtered to that target subset while preserving target order exactly.
     *
     * Semantics:
     * - Row order follows @p sources.
     * - Column order follows @p targets when provided.
     * - If @p targets is not provided, each row contains distances for all
     *   vertices in vertex-id order.
     * - Unreachable vertices retain the BFS unreachable sentinel value (`-1`).
     *
     * This overload operates on a prepared native CSR graph.
     *
     * @param graph Prepared native CSR graph.
     * @param sources Source vertices.
     * @param targets Optional target subset.
     * @param num_threads Requested thread count. If <= 1, execution is
     *   single-threaded. If 0, an implementation-defined default is used.
     *
     * @return Dense matrix of BFS distances.
     *
     * @throws std::out_of_range If any source or target vertex is out of range.
     */
    std::vector<std::vector<int> > multi_source_bfs_lengths(
        const NativeCsrGraph &graph,
        const std::vector<int> &sources,
        const std::optional<std::vector<int> > &targets = std::nullopt,
        int num_threads = 0
    );

    /**
     * @brief Compute weighted shortest-path lengths for multiple sources.
     *
     * Executes one SSSP traversal per source and assembles a dense distance matrix
     * whose row order matches @p sources exactly. If @p targets is provided, each
     * row is filtered to that target subset while preserving target order exactly.
     *
     * Semantics:
     * - Row order follows @p sources.
     * - Column order follows @p targets when provided.
     * - If @p targets is not provided, each row contains distances for all
     *   vertices in vertex-id order.
     * - Unreachable vertices retain the SSSP unreachable sentinel value (`inf`).
     *
     * This overload operates on a prepared native CSR graph.
     *
     * @param graph Prepared native CSR graph.
     * @param sources Source vertices.
     * @param targets Optional target subset.
     * @param num_threads Requested thread count. If <= 1, execution is
     *   single-threaded. If 0, an implementation-defined default is used.
     *
     * @return Dense matrix of weighted shortest-path distances.
     *
     * @throws std::out_of_range If any source or target vertex is out of range.
     * @throws std::invalid_argument If the graph contains invalid negative weights.
     */
    std::vector<std::vector<double> > multi_source_sssp_lengths(
        const NativeCsrGraph &graph,
        const std::vector<int> &sources,
        const std::optional<std::vector<int> > &targets = std::nullopt,
        int num_threads = 0
    );
} // namespace gpupath
