// file: cpp/tests/cuda_bfs_smoke.cpp

#include "gpupath/cuda_bfs.hpp"
#include "gpupath/cuda_csr_graph.hpp"

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
        } catch (const std::invalid_argument&) {
            threw = true;
        }

        assert(threw);
    }
} // namespace

int main() {
    test_unambiguous_bfs();
    std::cout << "cuda_bfs_smoke: unambiguous case passed\n";

    test_valid_tied_predecessor();
    std::cout << "cuda_bfs_smoke: tied-predecessor case passed\n";

    test_weighted_graph_rejected();
    std::cout << "cuda_bfs_smoke: weighted rejection case passed\n";

    std::cout << "cuda_bfs_smoke passed\n";
    return 0;
}