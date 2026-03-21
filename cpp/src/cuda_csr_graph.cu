// file: cuda_csr_graph.cu

#include "gpupath/cuda_csr_graph.hpp"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace gpupath {
    namespace {
        /* ============================================================================
         * CUDA Helpers
         * ========================================================================== */

        /**
         * @brief Check a CUDA runtime call and throw on failure.
         *
         * CUDA APIs return status codes instead of throwing exceptions. This helper
         * converts a failing CUDA status into a readable std::runtime_error.
         *
         * @param status CUDA runtime return code.
         * @param operation Human-readable operation name.
         *
         * @throws std::runtime_error If @p status is not cudaSuccess.
         */
        void throw_if_cuda_error(
            const cudaError_t status,
            const char *operation
        ) {
            if (status == cudaSuccess) {
                return;
            }

            std::ostringstream oss;
            oss << operation
                    << " failed: "
                    << cudaGetErrorName(status)
                    << " - "
                    << cudaGetErrorString(status);

            throw std::runtime_error(oss.str());
        }
    } // namespace

    /* ============================================================================
     * Constructors and Destructor
     * ========================================================================== */

    /**
     * @brief Construct an unweighted GPU-resident CSR graph.
     *
     * This constructor:
     * 1. validates the host CSR structure
     * 2. allocates device buffers
     * 3. copies host CSR arrays to device memory
     *
     * The host vectors are not stored. They are only used as upload inputs.
     */
    CudaCsrGraph::CudaCsrGraph(
        const std::size_t num_vertices,
        const std::vector<int> &indptr,
        const std::vector<int> &indices
    )
        : num_vertices_(num_vertices),
          num_edges_(indices.size()),
          is_weighted_(false) {
        validate(num_vertices, indptr, indices);

        throw_if_cuda_error(
            cudaMalloc(reinterpret_cast<void **>(&d_indptr_), indptr.size() * sizeof(int)),
            "cudaMalloc(d_indptr_)"
        );

        try {
            throw_if_cuda_error(
                cudaMalloc(reinterpret_cast<void **>(&d_indices_), indices.size() * sizeof(int)),
                "cudaMalloc(d_indices_)"
            );

            throw_if_cuda_error(
                cudaMemcpy(
                    d_indptr_,
                    indptr.data(),
                    indptr.size() * sizeof(int),
                    cudaMemcpyHostToDevice
                ),
                "cudaMemcpy(d_indptr_)"
            );

            if (!indices.empty()) {
                throw_if_cuda_error(
                    cudaMemcpy(
                        d_indices_,
                        indices.data(),
                        indices.size() * sizeof(int),
                        cudaMemcpyHostToDevice
                    ),
                    "cudaMemcpy(d_indices_)"
                );
            }
        } catch (...) {
            release();
            throw;
        }
    }

    /**
     * @brief Construct a weighted GPU-resident CSR graph.
     *
     * This constructor performs the same upload flow as the unweighted version,
     * plus allocation and upload of the weights array.
     */
    CudaCsrGraph::CudaCsrGraph(
        const std::size_t num_vertices,
        const std::vector<int> &indptr,
        const std::vector<int> &indices,
        const std::vector<double> &weights
    )
        : num_vertices_(num_vertices),
          num_edges_(indices.size()),
          is_weighted_(true) {
        validate(num_vertices, indptr, indices, weights);

        throw_if_cuda_error(
            cudaMalloc(reinterpret_cast<void **>(&d_indptr_), indptr.size() * sizeof(int)),
            "cudaMalloc(d_indptr_)"
        );

        try {
            throw_if_cuda_error(
                cudaMalloc(reinterpret_cast<void **>(&d_indices_), indices.size() * sizeof(int)),
                "cudaMalloc(d_indices_)"
            );

            throw_if_cuda_error(
                cudaMalloc(reinterpret_cast<void **>(&d_weights_), weights.size() * sizeof(double)),
                "cudaMalloc(d_weights_)"
            );

            throw_if_cuda_error(
                cudaMemcpy(
                    d_indptr_,
                    indptr.data(),
                    indptr.size() * sizeof(int),
                    cudaMemcpyHostToDevice
                ),
                "cudaMemcpy(d_indptr_)"
            );

            if (!indices.empty()) {
                throw_if_cuda_error(
                    cudaMemcpy(
                        d_indices_,
                        indices.data(),
                        indices.size() * sizeof(int),
                        cudaMemcpyHostToDevice
                    ),
                    "cudaMemcpy(d_indices_)"
                );
            }

            if (!weights.empty()) {
                throw_if_cuda_error(
                    cudaMemcpy(
                        d_weights_,
                        weights.data(),
                        weights.size() * sizeof(double),
                        cudaMemcpyHostToDevice
                    ),
                    "cudaMemcpy(d_weights_)"
                );
            }
        } catch (...) {
            release();
            throw;
        }
    }

    /**
     * @brief Destroy the graph and free owned device buffers.
     */
    CudaCsrGraph::~CudaCsrGraph() {
        release();
    }

    /**
     * @brief Move-construct by transferring device-buffer ownership.
     */
    CudaCsrGraph::CudaCsrGraph(CudaCsrGraph &&other) noexcept
        : num_vertices_(other.num_vertices_),
          num_edges_(other.num_edges_),
          is_weighted_(other.is_weighted_),
          d_indptr_(other.d_indptr_),
          d_indices_(other.d_indices_),
          d_weights_(other.d_weights_) {
        other.num_vertices_ = 0;
        other.num_edges_ = 0;
        other.is_weighted_ = false;
        other.d_indptr_ = nullptr;
        other.d_indices_ = nullptr;
        other.d_weights_ = nullptr;
    }

    /**
     * @brief Move-assign by releasing current buffers and taking ownership from other.
     */
    CudaCsrGraph &CudaCsrGraph::operator=(CudaCsrGraph &&other) noexcept {
        if (this == &other) {
            return *this;
        }

        release();

        num_vertices_ = other.num_vertices_;
        num_edges_ = other.num_edges_;
        is_weighted_ = other.is_weighted_;
        d_indptr_ = other.d_indptr_;
        d_indices_ = other.d_indices_;
        d_weights_ = other.d_weights_;

        other.num_vertices_ = 0;
        other.num_edges_ = 0;
        other.is_weighted_ = false;
        other.d_indptr_ = nullptr;
        other.d_indices_ = nullptr;
        other.d_weights_ = nullptr;

        return *this;
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
        return d_indptr_;
    }

    const int *CudaCsrGraph::indices_data() const noexcept {
        return d_indices_;
    }

    const double *CudaCsrGraph::weights_data() const noexcept {
        return d_weights_;
    }

    /* ============================================================================
     * Validation
     * ========================================================================== */

    /**
     * @brief Validate unweighted host-side CSR invariants before GPU upload.
     */
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

    /**
     * @brief Validate weighted host-side CSR invariants before GPU upload.
     */
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

    /* ============================================================================
     * Private Helpers
     * ========================================================================== */

    /**
     * @brief Free all owned device buffers and reset object state.
     *
     * Shared cleanup path for:
     * - destructor
     * - move assignment
     * - constructor failure recovery
     */
    void CudaCsrGraph::release() noexcept {
        if (d_weights_ != nullptr) {
            cudaFree(d_weights_);
            d_weights_ = nullptr;
        }

        if (d_indices_ != nullptr) {
            cudaFree(d_indices_);
            d_indices_ = nullptr;
        }

        if (d_indptr_ != nullptr) {
            cudaFree(d_indptr_);
            d_indptr_ = nullptr;
        }

        num_vertices_ = 0;
        num_edges_ = 0;
        is_weighted_ = false;
    }
} // namespace gpupath
