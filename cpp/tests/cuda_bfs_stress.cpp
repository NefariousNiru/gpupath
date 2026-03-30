// file: cpp/tests/cuda_bfs_stress.cpp

#include "gpupath/cuda_bfs.hpp"
#include "gpupath/cuda_csr_graph.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

/**
 * @brief Simple CPU BFS to act as the "Ground Truth".
 */
gpupath::BfsResult cpu_bfs_reference(const int num_vertices,
                                     const std::vector<int> &indptr,
                                     const std::vector<int> &indices,
                                     const int source) {
    gpupath::BfsResult res;
    res.distances.assign(num_vertices, -1);
    res.predecessors.assign(num_vertices, -1);

    std::queue<int> q;
    res.distances[source] = 0;
    q.push(source);

    while (!q.empty()) {
        const int u = q.front();
        q.pop();

        for (int i = indptr[u]; i < indptr[u + 1]; ++i) {
            if (int v = indices[i]; res.distances[v] == -1) {
                res.distances[v] = res.distances[u] + 1;
                res.predecessors[v] = u;
                q.push(v);
            }
        }
    }
    return res;
}

/**
 * @brief Generates a random graph with a "Hub" node to stress Warp Load Balancing.
 */
void run_stress_test(const int num_vertices, const int edges_per_vertex) {
    std::cout << "Generating graph: " << num_vertices << " vertices, ~"
            << num_vertices * edges_per_vertex << " edges...\n";

    std::vector<int> indptr(num_vertices + 1, 0);
    std::vector<int> indices;
    indices.reserve(num_vertices * edges_per_vertex);

    std::mt19937 rng(42);
    const std::uniform_int_distribution<int> dist(0, num_vertices - 1);

    // Create a "Power Law" style distribution: Node 0 is a massive hub
    for (int i = 0; i < num_vertices; ++i) {
        indptr[i] = static_cast<int>(indices.size());

        const int degree = (i == 0) ? (num_vertices / 10) : edges_per_vertex;
        for (int d = 0; d < degree; ++d) {
            indices.push_back(dist(rng));
        }
    }
    indptr[num_vertices] = static_cast<int>(indices.size());

    // Upload to GPU
    gpupath::CudaCsrGraph graph(num_vertices, indptr, indices);

    // --- GPU BFS ---
    const auto start_gpu = std::chrono::high_resolution_clock::now();
    gpupath::BfsResult gpu_res = gpupath::cuda_bfs_unweighted(graph, 0);
    const auto end_gpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> gpu_ms = end_gpu - start_gpu;
    std::cout << "GPU BFS took: " << gpu_ms.count() << " ms\n";

    // --- CPU BFS (Verification) ---
    std::cout << "Running CPU Reference for verification...\n";
    const auto start_cpu = std::chrono::high_resolution_clock::now();
    auto [distances, predecessors] = cpu_bfs_reference(num_vertices, indptr, indices, 0);
    const auto end_cpu = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double, std::milli> cpu_ms = end_cpu - start_cpu;
    std::cout << "CPU BFS took: " << cpu_ms.count() << " ms (Speedup: "
            << cpu_ms.count() / gpu_ms.count() << "x)\n";

    // --- Correctness Checks ---
    for (int i = 0; i < num_vertices; ++i) {
        if (gpu_res.distances[i] != distances[i]) {
            std::cerr << "Mismatch at vertex " << i << ": GPU dist=" << gpu_res.distances[i]
                    << ", CPU dist=" << distances[i] << "\n";
            exit(1);
        }
        // Note: Predecessors might differ in parallel BFS,
        // so we check if the GPU predecessor is a valid neighbor that leads to the distance.
        if (gpu_res.distances[i] > 0) {
            int p = gpu_res.predecessors[i];
            assert(p != -1);
            assert(gpu_res.distances[p] + 1 == gpu_res.distances[i]);
        }
    }
    std::cout << "STRESS TEST PASSED: Results verified.\n\n";
}

int main() {
    try {
        // Test 1: Small - 100k nodes
        run_stress_test(100'000, 20);

        // Test 2: Large (Stresses RTX 2060 VRAM)
        // ~1M nodes, ~50M edges. This uses ~200MB for indices,
        // plus buffers. Easily fits in 6GB but starts to show GPU throughput.
        run_stress_test(1'000'000, 50);

        // Test 3: The "Mega" Hub
        // One node connected to almost everyone else.
        // This forces your warp-expansion kernel to work hard on level 1.
        run_stress_test(2'000'000, 5);
    } catch (const std::exception &e) {
        std::cerr << "Test failed with error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
