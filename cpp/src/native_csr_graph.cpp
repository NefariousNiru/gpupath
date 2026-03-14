// file: native_csr_graph.cpp

#include "gpupath/native_csr_graph.hpp"

#include <sstream>
#include <stdexcept>
#include <string>

namespace gpupath {
    namespace {
        /* ============================================================================
         * Internal helpers
         * ========================================================================== */

        /**
         * @brief Build a consistent invalid_argument message.
         */
        [[nodiscard]] std::invalid_argument make_validation_error(const std::string &message) {
            return std::invalid_argument("NativeCsrGraph validation failed: " + message);
        }
    } // namespace

    /* ============================================================================
     * NativeCsrGraph: construction
     * ========================================================================== */

    NativeCsrGraph::NativeCsrGraph(
        const std::size_t num_vertices,
        std::vector<int> indptr,
        std::vector<int> indices
    )
        : num_vertices_(num_vertices),
          indptr_(std::move(indptr)),
          indices_(std::move(indices)) {
        validate();
    }

    NativeCsrGraph::NativeCsrGraph(
        const std::size_t num_vertices,
        std::vector<int> indptr,
        std::vector<int> indices,
        std::vector<double> weights
    )
        : num_vertices_(num_vertices),
          indptr_(std::move(indptr)),
          indices_(std::move(indices)),
          weights_(std::move(weights)) {
        validate();
    }

    /* ============================================================================
     * NativeCsrGraph: observers
     * ========================================================================== */

    std::size_t NativeCsrGraph::num_vertices() const noexcept {
        return num_vertices_;
    }

    std::size_t NativeCsrGraph::num_edges() const noexcept {
        return indices_.size();
    }

    bool NativeCsrGraph::is_weighted() const noexcept {
        return weights_.has_value();
    }

    const std::vector<int> &NativeCsrGraph::indptr() const noexcept {
        return indptr_;
    }

    const std::vector<int> &NativeCsrGraph::indices() const noexcept {
        return indices_;
    }

    const std::optional<std::vector<double> > &NativeCsrGraph::weights() const noexcept {
        return weights_;
    }

    /* ============================================================================
     * NativeCsrGraph: validation
     * ========================================================================== */

    void NativeCsrGraph::validate() const {
        if (indptr_.size() != num_vertices_ + 1U) {
            std::ostringstream oss;
            oss << "indptr length must be num_vertices + 1; got indptr.size()="
                    << indptr_.size() << ", num_vertices=" << num_vertices_;
            throw make_validation_error(oss.str());
        }

        if (indptr_.empty()) {
            throw make_validation_error("indptr must not be empty");
        }

        if (indptr_.front() != 0) {
            std::ostringstream oss;
            oss << "indptr[0] must be 0; got " << indptr_.front();
            throw make_validation_error(oss.str());
        }

        for (std::size_t i = 1; i < indptr_.size(); ++i) {
            if (indptr_[i] < indptr_[i - 1]) {
                std::ostringstream oss;
                oss << "indptr must be non-decreasing; indptr[" << (i - 1) << "]="
                        << indptr_[i - 1] << ", indptr[" << i << "]=" << indptr_[i];
                throw make_validation_error(oss.str());
            }
        }

        if (indptr_.back() < 0) {
            std::ostringstream oss;
            oss << "indptr.back() must be non-negative; got " << indptr_.back();
            throw make_validation_error(oss.str());
        }

        if (static_cast<std::size_t>(indptr_.back()) != indices_.size()) {
            std::ostringstream oss;
            oss << "indptr.back() must equal number of edges; got indptr.back()="
                    << indptr_.back() << ", indices.size()=" << indices_.size();
            throw make_validation_error(oss.str());
        }

        for (std::size_t edge_idx = 0; edge_idx < indices_.size(); ++edge_idx) {
            if (const int dst = indices_[edge_idx]; dst < 0 || static_cast<std::size_t>(dst) >= num_vertices_) {
                std::ostringstream oss;
                oss << "indices[" << edge_idx << "] out of range; got " << dst
                        << ", valid range=[0, " << num_vertices_ << ")";
                throw make_validation_error(oss.str());
            }
        }

        if (weights_.has_value() && weights_->size() != indices_.size()) {
            std::ostringstream oss;
            oss << "weights size must match indices size; got weights.size()="
                    << weights_->size() << ", indices.size()=" << indices_.size();
            throw make_validation_error(oss.str());
        }
    }
} // namespace gpupath
