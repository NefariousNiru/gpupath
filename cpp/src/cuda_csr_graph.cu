// file: cpp/src/cuda_csr_graph.cu

#include "gpupath/cuda_csr_graph.hpp"
#include "gpupath/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace gpupath {
    /* ============================================================================
     * Constructors
     * ========================================================================== */

    CudaCsrGraph::CudaCsrGraph(
        const std::size_t num_vertices,
        const std::vector<int> &indptr,
        const std::vector<int> &indices
    )
        : num_vertices_(num_vertices),
          num_edges_(indices.size()),
          is_weighted_(false),
          d_indptr_(indptr.size()),
          d_indices_(indices.size()),
          h_indptr_(indptr) {
        validate(num_vertices, indptr, indices);

        profile_ = build_graph_profile(num_vertices, indptr);

        cuda::throw_if_error(
            cudaMemcpy(
                d_indptr_.get(),
                indptr.data(),
                indptr.size() * sizeof(int),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy(d_indptr_)"
        );

        if (!indices.empty()) {
            cuda::throw_if_error(
                cudaMemcpy(
                    d_indices_.get(),
                    indices.data(),
                    indices.size() * sizeof(int),
                    cudaMemcpyHostToDevice
                ),
                "cudaMemcpy(d_indices_)"
            );
        }
    }

    CudaCsrGraph::CudaCsrGraph(
        const std::size_t num_vertices,
        const std::vector<int> &indptr,
        const std::vector<int> &indices,
        const std::vector<double> &weights
    )
        : num_vertices_(num_vertices),
          num_edges_(indices.size()),
          is_weighted_(true),
          d_indptr_(indptr.size()),
          d_indices_(indices.size()),
          d_weights_(weights.size()),
          h_indptr_(indptr) {
        validate(num_vertices, indptr, indices, weights);

        profile_ = build_graph_profile(num_vertices, indptr);

        cuda::throw_if_error(
            cudaMemcpy(
                d_indptr_.get(),
                indptr.data(),
                indptr.size() * sizeof(int),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy(d_indptr_)"
        );

        if (!indices.empty()) {
            cuda::throw_if_error(
                cudaMemcpy(
                    d_indices_.get(),
                    indices.data(),
                    indices.size() * sizeof(int),
                    cudaMemcpyHostToDevice
                ),
                "cudaMemcpy(d_indices_)"
            );
        }

        if (!weights.empty()) {
            cuda::throw_if_error(
                cudaMemcpy(
                    d_weights_.get(),
                    weights.data(),
                    weights.size() * sizeof(double),
                    cudaMemcpyHostToDevice
                ),
                "cudaMemcpy(d_weights_)"
            );
        }
    }

    /* ============================================================================
     * Observers
     * ========================================================================== */

    std::size_t CudaCsrGraph::num_vertices() const noexcept {
        return num_vertices_;
    }

    std::size_t CudaCsrGraph::num_edges() const noexcept {
        return num_edges_;
    }

    bool CudaCsrGraph::is_weighted() const noexcept {
        return is_weighted_;
    }

    const int *CudaCsrGraph::indptr_data() const noexcept {
        return d_indptr_.get();
    }

    const int *CudaCsrGraph::indices_data() const noexcept {
        return d_indices_.get();
    }

    const double *CudaCsrGraph::weights_data() const noexcept {
        return is_weighted_ ? d_weights_.get() : nullptr;
    }

    const GraphProfile &CudaCsrGraph::profile() const noexcept {
        return profile_;
    }

    const std::vector<int> &CudaCsrGraph::host_indptr() const noexcept {
        return h_indptr_;
    }

    /* ============================================================================
     * Validation
     * ========================================================================== */

    void CudaCsrGraph::validate(
        const std::size_t num_vertices,
        const std::vector<int> &indptr,
        const std::vector<int> &indices
    ) {
        if (indptr.size() != num_vertices + 1) {
            throw std::invalid_argument("indptr size must equal num_vertices + 1");
        }

        if (indptr.empty()) {
            throw std::invalid_argument("indptr must not be empty");
        }

        if (indptr.front() != 0) {
            throw std::invalid_argument("indptr[0] must be 0");
        }

        for (std::size_t i = 1; i < indptr.size(); ++i) {
            if (indptr[i] < indptr[i - 1]) {
                throw std::invalid_argument("indptr must be nondecreasing");
            }
        }

        if (indptr.back() < 0) {
            throw std::invalid_argument("indptr.back() must be nonnegative");
        }

        if (static_cast<std::size_t>(indptr.back()) != indices.size()) {
            throw std::invalid_argument("indices size must equal indptr.back()");
        }

        for (const int dst: indices) {
            if (dst < 0 || static_cast<std::size_t>(dst) >= num_vertices) {
                throw std::invalid_argument("indices contains out-of-range vertex id");
            }
        }
    }

    void CudaCsrGraph::validate(
        const std::size_t num_vertices,
        const std::vector<int> &indptr,
        const std::vector<int> &indices,
        const std::vector<double> &weights
    ) {
        validate(num_vertices, indptr, indices);

        if (weights.size() != indices.size()) {
            throw std::invalid_argument("weights size must equal indices size");
        }
    }
} // namespace gpupath
