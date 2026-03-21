// file: cuda_csr_graph.hpp

#pragma once

#include <cstddef>
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

        ~CudaCsrGraph();

        CudaCsrGraph(const CudaCsrGraph &) = delete;

        CudaCsrGraph(CudaCsrGraph &&other) noexcept;

        CudaCsrGraph &operator=(const CudaCsrGraph &) = delete;

        CudaCsrGraph &operator=(CudaCsrGraph &&other) noexcept;

        /* ------------------------------------------------------------------------
         * Observers
         * --------------------------------------------------------------------- */

        /**
         * @brief Return the number of vertices.
         */
        [[nodiscard]] std::size_t num_vertices() const noexcept;

        /**
         * @brief Return the number of edges.
         */
        [[nodiscard]] std::size_t num_edges() const noexcept;

        /**
         * @brief Return whether the graph is weighted.
         */
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

    private:
        /**
         * @brief Validate host-side CSR storage invariants for an unweighted graph.
         *
         * @throws std::invalid_argument If invariants are violated.
         */
        static void validate(
            std::size_t num_vertices,
            const std::vector<int> &indptr,
            const std::vector<int> &indices
        );

        /**
         * @brief Validate host-side CSR storage invariants for a weighted graph.
         *
         * @throws std::invalid_argument If invariants are violated.
         */
        static void validate(
            std::size_t num_vertices,
            const std::vector<int> &indptr,
            const std::vector<int> &indices,
            const std::vector<double> &weights
        );

        /**
         * @brief Release owned device buffers.
         */
        void release() noexcept;

    private:
        std::size_t num_vertices_{0};
        std::size_t num_edges_{0};
        bool is_weighted_{false};

        int *d_indptr_{nullptr};
        int *d_indices_{nullptr};
        double *d_weights_{nullptr};
    };
} // namespace gpupath
