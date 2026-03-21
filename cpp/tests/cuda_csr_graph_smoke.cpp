// file: cpp/tests/cuda_csr_graph_smoke.cpp

#include "gpupath/cuda_csr_graph.hpp"

#include <cassert>
#include <iostream>
#include <vector>

/**
 * @brief Minimal smoke test for CUDA prepared CSR graph construction.
 *
 * This test verifies only:
 * - unweighted construction succeeds
 * - weighted construction succeeds
 * - graph metadata is reported correctly
 *
 * It does not launch any kernels yet.
 */
int main() {
    {
        const gpupath::CudaCsrGraph graph(
            4,
            std::vector<int>{0, 2, 3, 5, 5},
            std::vector<int>{1, 2, 2, 0, 3}
        );

        assert(graph.num_vertices() == 4);
        assert(graph.num_edges() == 5);
        assert(graph.is_weighted() == false);
        assert(graph.indptr_data() != nullptr);
        assert(graph.indices_data() != nullptr);
        assert(graph.weights_data() == nullptr);

        std::cout << "unweighted cuda csr smoke passed\n";
    }

    {
        const gpupath::CudaCsrGraph graph(
            4,
            std::vector<int>{0, 2, 3, 5, 5},
            std::vector<int>{1, 2, 2, 0, 3},
            std::vector<double>{1.0, 2.5, 0.5, 3.0, 4.0}
        );

        assert(graph.num_vertices() == 4);
        assert(graph.num_edges() == 5);
        assert(graph.is_weighted() == true);
        assert(graph.indptr_data() != nullptr);
        assert(graph.indices_data() != nullptr);
        assert(graph.weights_data() != nullptr);

        std::cout << "weighted cuda csr smoke passed\n";
    }

    std::cout << "cuda_csr_graph_smoke passed\n";
    return 0;
}
