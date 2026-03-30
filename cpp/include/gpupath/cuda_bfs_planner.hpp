// file: cpp/include/gpupath/cuda_bfs_planner.hpp

#pragma once

#include "gpupath/cuda_graph_profile.hpp"

namespace gpupath {
    /* ============================================================================
     * BFS Planner Types
     * ========================================================================== */

    enum class TraversalPolicy {
        ThreadMapped,
        WarpMapped,
        HybridMapped
    };

    enum class FrontierPolicy {
        GlobalAtomicQueue,
        WarpAggregatedQueue,
        BlockAggregatedQueue
    };

    struct FrontierStats {
        int level{0};
        int frontier_size{0};
        int next_frontier_size{0};

        std::size_t frontier_edges_scanned{0};
        double average_frontier_degree{0.0};
        int max_frontier_degree{0};

        double visited_fraction{0.0};
        double discovery_efficiency{0.0};
    };

    struct BfsExecutionPlan {
        TraversalPolicy traversal{TraversalPolicy::WarpMapped};
        FrontierPolicy frontier{FrontierPolicy::GlobalAtomicQueue};
    };

    [[nodiscard]] inline const char *to_string(const TraversalPolicy policy) noexcept {
        switch (policy) {
            case TraversalPolicy::ThreadMapped:
                return "ThreadMapped";
            case TraversalPolicy::WarpMapped:
                return "WarpMapped";
            case TraversalPolicy::HybridMapped:
                return "HybridMapped";
        }
        return "UnknownTraversalPolicy";
    }

    [[nodiscard]] inline const char *to_string(const FrontierPolicy policy) noexcept {
        switch (policy) {
            case FrontierPolicy::GlobalAtomicQueue:
                return "GlobalAtomicQueue";
            case FrontierPolicy::WarpAggregatedQueue:
                return "WarpAggregatedQueue";
            case FrontierPolicy::BlockAggregatedQueue:
                return "BlockAggregatedQueue";
        }
        return "UnknownFrontierPolicy";
    }

    /**
     * @brief Factory function to generate real-time BFS frontier statistics.
     *
     * This function creates a snapshot of the current BFS level's characteristics.
     * It acts as the primary input for the Execution Planner, allowing it to
     * dynamically switch between expansion strategies (e.g., Thread-per-Vertex
     * vs. Warp-per-Vertex).
     *
     * @note Many fields are currently "Bootstrap Estimates." In a production
     * system, these would be refined by sampling the actual frontier vertices
     * on the GPU or using analytical models based on the @p graph_profile.
     *
     * @param graph_profile The static structural analysis of the entire graph.
     * @param level The current depth of the BFS (0 = source).
     * @param frontier_size The number of active vertices to be expanded in this level.
     *
     * @return [[nodiscard]] A FrontierStats object containing heuristic estimates
     * of the workload, skew, and density for the current level.
     */
    [[nodiscard]] inline FrontierStats make_frontier_stats(
        const GraphProfile &graph_profile,
        const int level,
        const int frontier_size
    ) {
        FrontierStats stats{};
        stats.level = level;
        stats.frontier_size = frontier_size;
        stats.next_frontier_size = 0;

        if (frontier_size <= 0) {
            stats.frontier_edges_scanned = 0;
            stats.average_frontier_degree = 0.0;
            stats.max_frontier_degree = 0;
            stats.visited_fraction = 0.0;
            stats.discovery_efficiency = 0.0;
            return stats;
        }

        // Bootstrap estimate only:
        // we do not yet measure the actual current frontier composition.
        stats.average_frontier_degree = graph_profile.average_degree;

        stats.frontier_edges_scanned = static_cast<std::size_t>(
            static_cast<double>(frontier_size) * stats.average_frontier_degree
        );

        // Be conservative but not absurdly pessimistic.
        // Using global max degree at every level can overstate skew badly.
        stats.max_frontier_degree = std::max(
            static_cast<int>(graph_profile.degree_p90),
            static_cast<int>(graph_profile.degree_p99)
        );

        stats.visited_fraction = 0.0;
        stats.discovery_efficiency = 0.0;
        return stats;
    }


    /**
     * @brief Choose a simple BFS execution plan from static and frontier signals.
     *
     * This is intentionally heuristic and conservative for the first planner pass.
     */
    [[nodiscard]] inline BfsExecutionPlan choose_bfs_plan(
        const GraphProfile &graph_profile,
        const FrontierStats &frontier_stats
    ) {
        BfsExecutionPlan plan{};

        const bool tiny_frontier = frontier_stats.frontier_size <= 32;
        const bool small_frontier = frontier_stats.frontier_size <= 128;
        const bool low_frontier_degree = frontier_stats.average_frontier_degree <= 4.0;
        const bool moderate_frontier_degree = frontier_stats.average_frontier_degree <= 16.0;
        const bool extreme_frontier_skew =
                frontier_stats.max_frontier_degree >= 8 * (graph_profile.degree_p90 + 1);

        if (tiny_frontier) {
            plan.traversal = TraversalPolicy::ThreadMapped;
            plan.frontier = FrontierPolicy::GlobalAtomicQueue;
            return plan;
        }

        if (low_frontier_degree && !extreme_frontier_skew) {
            plan.traversal = TraversalPolicy::ThreadMapped;
            plan.frontier = FrontierPolicy::GlobalAtomicQueue;
            return plan;
        }

        if (small_frontier && moderate_frontier_degree && !extreme_frontier_skew) {
            plan.traversal = TraversalPolicy::ThreadMapped;
            plan.frontier = FrontierPolicy::GlobalAtomicQueue;
            return plan;
        }

        plan.traversal = TraversalPolicy::WarpMapped;
        plan.frontier = FrontierPolicy::GlobalAtomicQueue;
        return plan;
    }
} // namespace gpupath
