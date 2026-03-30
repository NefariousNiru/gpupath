// file: cpp/include/gpupath/cuda_csr_graph.hpp

#pragma once

#include "gpupath/cuda_utils.hpp"
#include "gpupath/cuda_graph_profile.hpp"

#include <vector>

namespace gpupath {
    /* ============================================================================
     * CudaCsrGraph
     * ========================================================================== */

    /**
     * @brief CUDA immutable CSR graph storage.
     *
     * Owns CSR adjacency arrays in device memory. This class is the prepared/CUDA
     * representation used to amortize host -> device graph transfer costs across
     * repeated graph queries on a static graph.
     */
    class CudaCsrGraph {
    public:
        /**
         * @brief Construct an unweighted CUDA CSR graph.
         *
         * @param num_vertices Number of vertices in the graph.
         * @param indptr CSR row-pointer array of length num_vertices + 1.
         * @param indices CSR column-index array of length num_edges.
         *
         * @throws std::invalid_argument If CSR invariants are violated.
         * @throws std::runtime_error If CUDA allocation or transfer fails.
         */
        CudaCsrGraph(
            std::size_t num_vertices,
            const std::vector<int> &indptr,
            const std::vector<int> &indices
        );

        /**
         * @brief Construct a weighted CUDA CSR graph.
         *
         * @param num_vertices Number of vertices in the graph.
         * @param indptr CSR row-pointer array of length num_vertices + 1.
         * @param indices CSR column-index array of length num_edges.
         * @param weights Edge weights parallel to @p indices.
         *
         * @throws std::invalid_argument If CSR invariants are violated.
         * @throws std::runtime_error If CUDA allocation or transfer fails.
         */
        CudaCsrGraph(
            std::size_t num_vertices,
            const std::vector<int> &indptr,
            const std::vector<int> &indices,
            const std::vector<double> &weights
        );

        ~CudaCsrGraph() = default;

        CudaCsrGraph(const CudaCsrGraph &) = delete;

        CudaCsrGraph &operator=(const CudaCsrGraph &) = delete;

        CudaCsrGraph(CudaCsrGraph &&) noexcept = default;

        CudaCsrGraph &operator=(CudaCsrGraph &&) noexcept = default;

        /* ------------------------------------------------------------------------
         * Observers
         * --------------------------------------------------------------------- */

        [[nodiscard]] std::size_t num_vertices() const noexcept;

        [[nodiscard]] std::size_t num_edges() const noexcept;

        [[nodiscard]] bool is_weighted() const noexcept;

        /**
         * @brief Return the device CSR row-pointer buffer.
         *
         * Internal/native use only.
         */
        [[nodiscard]] const int *indptr_data() const noexcept;

        /**
         * @brief Return the device CSR column-index buffer.
         *
         * Internal/native use only.
         */
        [[nodiscard]] const int *indices_data() const noexcept;

        /**
         * @brief Return the device weights buffer.
         *
         * Returns nullptr for unweighted graphs.
         * Internal/native use only.
         */
        [[nodiscard]] const double *weights_data() const noexcept;

        /**
         * @brief Return the static host-side graph profile used by planners.
         */
        [[nodiscard]] const GraphProfile &profile() const noexcept;

        /**
         * @brief Return the host CSR row-pointer cache.
         *
         * Internal/planner use only. This exists to support frontier profiling
         * without copying the full CSR structure back from device memory.
         */
        [[nodiscard]] const std::vector<int> &host_indptr() const noexcept;

    private:
        static void validate(
            std::size_t num_vertices,
            const std::vector<int> &indptr,
            const std::vector<int> &indices
        );

        static void validate(
            std::size_t num_vertices,
            const std::vector<int> &indptr,
            const std::vector<int> &indices,
            const std::vector<double> &weights
        );

    private:
        std::size_t num_vertices_{0};
        std::size_t num_edges_{0};
        bool is_weighted_{false};

        cuda::DeviceBuffer<int> d_indptr_;
        cuda::DeviceBuffer<int> d_indices_;
        cuda::DeviceBuffer<double> d_weights_;

        std::vector<int> h_indptr_;
        GraphProfile profile_{};
    };
} // namespace gpupath
