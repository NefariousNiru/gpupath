// file: cpp/include/gpupath/cuda_graph_profile.hpp

#pragma once

#include <algorithm>
#include <vector>

namespace gpupath {
    /* ============================================================================
     * Graph Profile
     * ========================================================================== */

    /**
     * @brief Static degree/profile summary for a prepared CUDA CSR graph.
     *
     * This is computed once from the host CSR structure at graph preparation time
     * and then reused by the BFS planner across repeated traversals.
     */
    struct GraphProfile {
        std::size_t num_vertices{0};
        std::size_t num_edges{0};

        double average_degree{0.0};
        int max_degree{0};

        int degree_p50{0};
        int degree_p90{0};
        int degree_p99{0};

        std::size_t bucket_deg_0_2{0};
        std::size_t bucket_deg_3_8{0};
        std::size_t bucket_deg_9_32{0};
        std::size_t bucket_deg_33_256{0};
        std::size_t bucket_deg_257_plus{0};
    };

    namespace detail {
        [[nodiscard]] inline int percentile_from_sorted_degrees(
            const std::vector<int> &sorted_degrees,
            const double q
        ) {
            if (sorted_degrees.empty()) {
                return 0;
            }

            const double clamped_q = std::clamp(q, 0.0, 1.0);
            const auto idx = static_cast<std::size_t>(
                clamped_q * static_cast<double>(sorted_degrees.size() - 1)
            );
            return sorted_degrees[idx];
        }
    } // namespace detail

    /**
     * @brief Build a static graph profile from a valid CSR row-pointer array.
     *
     * Assumes @p indptr is already structurally valid for @p num_vertices.
     */
    [[nodiscard]] inline GraphProfile build_graph_profile(
        const std::size_t num_vertices,
        const std::vector<int> &indptr
    ) {
        GraphProfile profile{};
        profile.num_vertices = num_vertices;
        profile.num_edges = indptr.empty() ? 0 : static_cast<std::size_t>(indptr.back());

        if (num_vertices == 0) {
            return profile;
        }

        std::vector<int> degrees;
        degrees.reserve(num_vertices);

        std::size_t degree_sum = 0;
        int max_degree = 0;

        for (std::size_t u = 0; u < num_vertices; ++u) {
            const int degree = indptr[u + 1] - indptr[u];
            degrees.push_back(degree);

            degree_sum += static_cast<std::size_t>(degree);
            max_degree = std::max(max_degree, degree);

            if (degree <= 2) {
                ++profile.bucket_deg_0_2;
            } else if (degree <= 8) {
                ++profile.bucket_deg_3_8;
            } else if (degree <= 32) {
                ++profile.bucket_deg_9_32;
            } else if (degree <= 256) {
                ++profile.bucket_deg_33_256;
            } else {
                ++profile.bucket_deg_257_plus;
            }
        }

        std::sort(degrees.begin(), degrees.end());

        profile.average_degree =
                static_cast<double>(degree_sum) / static_cast<double>(num_vertices);
        profile.max_degree = max_degree;
        profile.degree_p50 = detail::percentile_from_sorted_degrees(degrees, 0.50);
        profile.degree_p90 = detail::percentile_from_sorted_degrees(degrees, 0.90);
        profile.degree_p99 = detail::percentile_from_sorted_degrees(degrees, 0.99);

        return profile;
    }
} // namespace gpupath
