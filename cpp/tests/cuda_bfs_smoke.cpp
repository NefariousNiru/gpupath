// file: cpp/tests/cuda_bfs_smoke.cpp

#include "gpupath/cuda_bfs.hpp"
#include "gpupath/cuda_bfs_planner.hpp"
#include "gpupath/cuda_csr_graph.hpp"
#include "gpupath/cuda_graph_profile.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
    void test_unambiguous_bfs() {
        // Graph:
        //   0 -> 1, 2
        //   1 -> 3
        //   2 -> 4
        //   4 -> 5
        //   6 disconnected
        //
        // Distances from 0:
        //   [0, 1, 1, 2, 2, 3, -1]
        //
        // Predecessors are unambiguous here:
        //   [-1, 0, 0, 1, 2, 4, -1]

        const gpupath::CudaCsrGraph graph(
            7,
            std::vector<int>{0, 2, 3, 4, 4, 5, 5, 5},
            std::vector<int>{1, 2, 3, 4, 5}
        );

        const gpupath::BfsResult result = gpupath::cuda_bfs_unweighted(graph, 0);

        const std::vector<int> expected_distances{0, 1, 1, 2, 2, 3, -1};
        const std::vector<int> expected_predecessors{-1, 0, 0, 1, 2, 4, -1};

        assert(result.distances == expected_distances);
        assert(result.predecessors == expected_predecessors);
    }

    void test_valid_tied_predecessor() {
        // Graph:
        //   0 -> 1, 2
        //   1 -> 3
        //   2 -> 3, 4
        //   4 -> 5
        //
        // Distances from 0:
        //   [0, 1, 1, 2, 2, 3]
        //
        // Vertex 3 has two valid same-level parents: 1 or 2.
        // Parallel BFS may legally choose either.

        const gpupath::CudaCsrGraph graph(
            6,
            std::vector<int>{0, 2, 3, 5, 5, 6, 6},
            std::vector<int>{1, 2, 3, 3, 4, 5}
        );

        const gpupath::BfsResult result = gpupath::cuda_bfs_unweighted(graph, 0);

        const std::vector<int> expected_distances{0, 1, 1, 2, 2, 3};
        assert(result.distances == expected_distances);

        assert(result.predecessors[0] == -1);
        assert(result.predecessors[1] == 0);
        assert(result.predecessors[2] == 0);
        assert(result.predecessors[4] == 2);
        assert(result.predecessors[5] == 4);

        // Tied predecessor: either 1 or 2 is valid for vertex 3.
        assert(result.predecessors[3] == 1 || result.predecessors[3] == 2);

        // Generic shortest-path parent invariant checks.
        for (std::size_t v = 1; v < result.distances.size(); ++v) {
            if (result.distances[v] == -1) {
                assert(result.predecessors[v] == -1);
                continue;
            }

            const int p = result.predecessors[v];
            assert(p >= 0);
            assert(static_cast<std::size_t>(p) < result.distances.size());
            assert(result.distances[p] + 1 == result.distances[v]);
        }
    }

    void test_weighted_graph_rejected() {
        const gpupath::CudaCsrGraph graph(
            3,
            std::vector<int>{0, 1, 2, 2},
            std::vector<int>{1, 2},
            std::vector<double>{1.0, 2.0}
        );

        bool threw = false;
        try {
            [[maybe_unused]] const gpupath::BfsResult result =
                gpupath::cuda_bfs_unweighted(graph, 0);
        } catch (const std::invalid_argument &) {
            threw = true;
        }

        assert(threw);
    }

    void test_planner_selects_thread_mapped_for_small_low_degree_frontier() {
        const gpupath::GraphProfile graph_profile{
            1000,   // num_vertices
            3000,   // num_edges
            3.0,    // average_degree
            12,     // max_degree
            2,      // degree_p50
            5,      // degree_p90
            8,      // degree_p99
            500,    // bucket_deg_0_2
            400,    // bucket_deg_3_8
            100,    // bucket_deg_9_32
            0,      // bucket_deg_33_256
            0       // bucket_deg_257_plus
        };

        const gpupath::FrontierStats frontier_stats{
            2,      // level
            24,     // frontier_size
            0,      // next_frontier_size
            72,     // frontier_edges_scanned
            3.0,    // average_frontier_degree
            8,      // max_frontier_degree
            0.0,    // visited_fraction
            0.0     // discovery_efficiency
        };

        const gpupath::BfsExecutionPlan plan =
            gpupath::choose_bfs_plan(graph_profile, frontier_stats);

        assert(plan.traversal == gpupath::TraversalPolicy::ThreadMapped);
        assert(plan.frontier == gpupath::FrontierPolicy::GlobalAtomicQueue);
    }

    void test_planner_selects_warp_mapped_for_large_high_degree_frontier() {
        const gpupath::GraphProfile graph_profile{
            100000, // num_vertices
            5000000,// num_edges
            50.0,   // average_degree
            4096,   // max_degree
            12,     // degree_p50
            96,     // degree_p90
            512,    // degree_p99
            1000,   // bucket_deg_0_2
            5000,   // bucket_deg_3_8
            25000,  // bucket_deg_9_32
            50000,  // bucket_deg_33_256
            19000   // bucket_deg_257_plus
        };

        const gpupath::FrontierStats frontier_stats{
            5,       // level
            5000,    // frontier_size
            0,       // next_frontier_size
            250000,  // frontier_edges_scanned
            50.0,    // average_frontier_degree
            2048,    // max_frontier_degree
            0.0,     // visited_fraction
            0.0      // discovery_efficiency
        };

        const gpupath::BfsExecutionPlan plan =
            gpupath::choose_bfs_plan(graph_profile, frontier_stats);

        assert(plan.traversal == gpupath::TraversalPolicy::WarpMapped);
        assert(plan.frontier == gpupath::FrontierPolicy::GlobalAtomicQueue);
    }

    void test_planner_selects_warp_mapped_for_extreme_skew() {
        const gpupath::GraphProfile graph_profile{
            10000,   // num_vertices
            200000,  // num_edges
            20.0,    // average_degree
            5000,    // max_degree
            4,       // degree_p50
            16,      // degree_p90
            128,     // degree_p99
            2000,    // bucket_deg_0_2
            4000,    // bucket_deg_3_8
            2500,    // bucket_deg_9_32
            1200,    // bucket_deg_33_256
            300      // bucket_deg_257_plus
        };

        const gpupath::FrontierStats frontier_stats{
            3,      // level
            64,     // frontier_size
            0,      // next_frontier_size
            256,    // frontier_edges_scanned
            4.0,    // average_frontier_degree
            4096,   // max_frontier_degree
            0.0,    // visited_fraction
            0.0     // discovery_efficiency
        };

        const gpupath::BfsExecutionPlan plan =
            gpupath::choose_bfs_plan(graph_profile, frontier_stats);

        assert(plan.traversal == gpupath::TraversalPolicy::WarpMapped);
        assert(plan.frontier == gpupath::FrontierPolicy::GlobalAtomicQueue);
    }
} // namespace

int main() {
    test_unambiguous_bfs();
    std::cout << "cuda_bfs_smoke: unambiguous case passed\n";

    test_valid_tied_predecessor();
    std::cout << "cuda_bfs_smoke: tied-predecessor case passed\n";

    test_weighted_graph_rejected();
    std::cout << "cuda_bfs_smoke: weighted rejection case passed\n";

    test_planner_selects_thread_mapped_for_small_low_degree_frontier();
    std::cout << "cuda_bfs_smoke: planner thread-mapped case passed\n";

    test_planner_selects_warp_mapped_for_large_high_degree_frontier();
    std::cout << "cuda_bfs_smoke: planner warp-mapped large/high-degree case passed\n";

    test_planner_selects_warp_mapped_for_extreme_skew();
    std::cout << "cuda_bfs_smoke: planner warp-mapped skew case passed\n";

    std::cout << "cuda_bfs_smoke passed\n";
    return 0;
}