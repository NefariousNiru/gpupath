// file: native_csr_graph.hpp

#pragma once

#include <optional>
#include <vector>

namespace gpupath {
    /* ============================================================================
     * NativeCsrGraph
     * ========================================================================== */

    /**
     * @brief Native immutable CSR graph storage.
     *
     * Owns CSR adjacency arrays in native memory. This class is the prepared/native
     * representation used to amortize Python -> C++ boundary costs across repeated
     * graph queries on a static graph.
     */
    class NativeCsrGraph {
    public:
        /**
         * @brief Construct an unweighted native CSR graph.
         *
         * @param num_vertices Number of vertices in the graph.
         * @param indptr CSR row-pointer array of length num_vertices + 1.
         * @param indices CSR column-index array of length num_edges.
         *
         * @throws std::invalid_argument If CSR invariants are violated.
         */
        NativeCsrGraph(
            std::size_t num_vertices,
            std::vector<int> indptr,
            std::vector<int> indices
        );

        /**
         * @brief Construct a weighted native CSR graph.
         *
         * @param num_vertices Number of vertices in the graph.
         * @param indptr CSR row-pointer array of length num_vertices + 1.
         * @param indices CSR column-index array of length num_edges.
         * @param weights Edge weights parallel to @p indices.
         *
         * @throws std::invalid_argument If CSR invariants are violated.
         */
        NativeCsrGraph(
            std::size_t num_vertices,
            std::vector<int> indptr,
            std::vector<int> indices,
            std::vector<double> weights
        );

        ~NativeCsrGraph() = default;

        NativeCsrGraph(const NativeCsrGraph &) = default;

        NativeCsrGraph(NativeCsrGraph &&) noexcept = default;

        NativeCsrGraph &operator=(const NativeCsrGraph &) = default;

        NativeCsrGraph &operator=(NativeCsrGraph &&) noexcept = default;

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
         * @brief Return the CSR row-pointer array.
         */
        [[nodiscard]] const std::vector<int> &indptr() const noexcept;

        /**
         * @brief Return the CSR column-index array.
         */
        [[nodiscard]] const std::vector<int> &indices() const noexcept;

        /**
         * @brief Return the optional weights array.
         *
         * Empty for unweighted graphs.
         */
        [[nodiscard]] const std::optional<std::vector<double> > &weights() const noexcept;

    private:
        /**
         * @brief Validate CSR storage invariants.
         *
         * Called exactly once from constructors after member ownership is
         * established.
         *
         * @throws std::invalid_argument If invariants are violated.
         */
        void validate() const;

    private:
        std::size_t num_vertices_{0};
        std::vector<int> indptr_{};
        std::vector<int> indices_{};
        std::optional<std::vector<double> > weights_{std::nullopt};
    };
} // namespace gpupath
