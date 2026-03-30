// file: cpp/src/cuda_bfs.cu

#include "gpupath/cuda_bfs.hpp"
#include "gpupath/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <stdexcept>


namespace gpupath {
    namespace {
        using cuda::DeviceBuffer;

        /* ============================================================================
         * Kernel Configuration
         * ========================================================================== */

        /**
         * @brief Logical warp size used for cooperative frontier expansion.
         *
         * The warp-expansion kernel assigns exactly one logical warp to one
         * frontier vertex. Each lane in the warp cooperates on the adjacency list
         * of that vertex by striding through outgoing edges in steps of
         * @ref kWarpSize.
         */
        constexpr int kWarpSize = 32;

        /**
         * @brief Number of CUDA threads per block for BFS kernels.
         *
         * This must be a multiple of @ref kWarpSize because the expansion kernel
         * derives warp ids and lane ids directly from the linear thread id.
         */
        constexpr int kThreadsPerBlock = 256;

        static_assert(
            kThreadsPerBlock % kWarpSize == 0,
            "kThreadsPerBlock must be a multiple of kWarpSize"
        );

        /* ============================================================================
         * Kernels
         * ========================================================================== */

        /**
         * @brief Initialize BFS output/state buffers on device.
         *
         * This kernel performs one-time initialization for a single-source BFS:
         * - all distances are initialized to -1
         * - all predecessors are initialized to -1
         * - the source distance is set to 0
         * - the first frontier contains only the source
         *
         * The host tracks frontier sizes across levels, so this kernel does not
         * initialize a device-side current-frontier count.
         *
         * @param num_vertices Number of vertices in the graph.
         * @param source Source vertex id.
         * @param distances Device output array of length @p num_vertices.
         * @param predecessors Device output array of length @p num_vertices.
         * @param curr_frontier Device frontier buffer whose first slot is seeded
         *        with @p source.
         */
        __global__ void bfs_init_kernel(
            const int num_vertices,
            const int source,
            int *distances,
            int *predecessors,
            int *curr_frontier
        ) {
            const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

            if (tid < num_vertices) {
                distances[tid] = -1;
                predecessors[tid] = -1;
            }

            if (tid == 0) {
                distances[source] = 0;
                predecessors[source] = -1;
                curr_frontier[0] = source;
            }
        }

        /**
         * @brief Expand one BFS level using warp-cooperative frontier processing.
         *
         * Mapping strategy:
         * - one logical warp is assigned to one frontier vertex
         * - `warp_id` selects the frontier vertex
         * - `lane_id` selects which subset of outgoing edges that lane will process
         *
         * For a frontier vertex @p u with adjacency range `[start, end)`, lane
         * `lane_id` processes:
         *
         * `start + lane_id, start + lane_id + kWarpSize, ...`
         *
         * This reduces the "one thread gets stuck on one high-degree vertex"
         * problem of the naive frontier kernel by letting 32 threads cooperate on
         * the same adjacency list.
         *
         * Discovery semantics remain the same as the baseline BFS kernel:
         * - the first thread to change `distances[v]` from -1 to `level + 1`
         *   wins discovery of vertex @p v
         * - the winning thread records `predecessors[v] = u`
         * - the winning thread appends @p v into @p next_frontier
         *
         * Notes:
         * - predecessor choices may differ from serial CPU BFS when multiple
         *   same-level parents are valid
         * - this kernel is not guaranteed to outperform the naive version on
         *   very low-degree graphs
         *
         * @param indptr Device CSR row-pointer array.
         * @param indices Device CSR column-index array.
         * @param curr_frontier Device buffer containing the current frontier.
         * @param curr_count Number of vertices in the current frontier.
         * @param next_frontier Device output buffer for the next frontier.
         * @param next_count Device counter for the next frontier size.
         * @param distances Device BFS distance array.
         * @param predecessors Device BFS predecessor array.
         * @param level Current BFS level being expanded.
         */
        __global__ void bfs_expand_warp_kernel(
            const int *indptr,
            const int *indices,
            const int *curr_frontier,
            const int curr_count,
            int *next_frontier,
            int *next_count,
            int *distances,
            int *predecessors,
            const int level
        ) {
            const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int warp_id = tid / kWarpSize;
            const int lane_id = tid % kWarpSize;

            if (warp_id >= curr_count) {
                return;
            }

            const int u = curr_frontier[warp_id];
            const int start = indptr[u];
            const int end = indptr[u + 1];

            for (int edge_idx = start + lane_id; edge_idx < end; edge_idx += kWarpSize) {
                if (const int v = indices[edge_idx]; atomicCAS(&distances[v], -1, level + 1) == -1) {
                    predecessors[v] = u;

                    const int slot = atomicAdd(next_count, 1);
                    next_frontier[slot] = v;
                }
            }
        }
    } // namespace

    BfsResult cuda_bfs_unweighted(const CudaCsrGraph &graph, const int source) {
        const std::size_t n_vertices = graph.num_vertices();

        if (source < 0 || static_cast<std::size_t>(source) >= n_vertices) {
            throw std::out_of_range("source out of range");
        }

        if (graph.is_weighted()) {
            throw std::invalid_argument("cuda_bfs_unweighted requires an unweighted graph");
        }

        const int n = static_cast<int>(n_vertices);

        DeviceBuffer<int> d_distances(n_vertices);
        DeviceBuffer<int> d_predecessors(n_vertices);
        DeviceBuffer<int> d_curr_frontier(n_vertices);
        DeviceBuffer<int> d_next_frontier(n_vertices);
        DeviceBuffer<int> d_next_count(1);

        {
            const int blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;

            bfs_init_kernel<<<blocks, kThreadsPerBlock>>>(
                n,
                source,
                d_distances.get(),
                d_predecessors.get(),
                d_curr_frontier.get()
            );

            cuda::throw_if_last_error("bfs_init_kernel launch");
            cuda::synchronize_or_throw("bfs_init_kernel synchronize");
        }

        int h_curr_count = 1;
        int level = 0;

        while (h_curr_count > 0) {
            cuda::throw_if_error(
                cudaMemsetAsync(d_next_count.get(), 0, sizeof(int)),
                "cudaMemsetAsync(next_count)"
            );

            const int total_threads = h_curr_count * kWarpSize;
            const int blocks = (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock;

            bfs_expand_warp_kernel<<<blocks, kThreadsPerBlock>>>(
                graph.indptr_data(),
                graph.indices_data(),
                d_curr_frontier.get(),
                h_curr_count,
                d_next_frontier.get(),
                d_next_count.get(),
                d_distances.get(),
                d_predecessors.get(),
                level
            );

            cuda::throw_if_last_error("bfs_expand_warp_kernel launch");

            int h_next_count = 0;
            cuda::throw_if_error(
                cudaMemcpy(
                    &h_next_count,
                    d_next_count.get(),
                    sizeof(int),
                    cudaMemcpyDeviceToHost
                ),
                "cudaMemcpy(next_count D2H)"
            );

            std::swap(d_curr_frontier, d_next_frontier);
            h_curr_count = h_next_count;
            ++level;
        }

        BfsResult result;
        result.distances.resize(n_vertices);
        result.predecessors.resize(n_vertices);

        cuda::throw_if_error(
            cudaMemcpy(
                result.distances.data(),
                d_distances.get(),
                n_vertices * sizeof(int),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy(distances D2H)"
        );

        cuda::throw_if_error(
            cudaMemcpy(
                result.predecessors.data(),
                d_predecessors.get(),
                n_vertices * sizeof(int),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy(predecessors D2H)"
        );

        return result;
    }
} // namespace gpupath
