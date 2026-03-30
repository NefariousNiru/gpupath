// file: cpp/include/gpupath/cuda_utils.hpp

#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <utility>

namespace gpupath::cuda {
    /**
     * @brief Check a CUDA runtime call and throw on failure.
     *
     * CUDA runtime APIs return status codes rather than throwing exceptions.
     * This helper converts a failing CUDA status into a readable
     * std::runtime_error.
     *
     * @param status CUDA runtime return code.
     * @param operation Human-readable operation name.
     *
     * @throws std::runtime_error If @p status is not cudaSuccess.
     */
    inline void throw_if_error(
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

    /**
     * @brief Check the most recent CUDA error and throw if one is present.
     *
     * Intended for use immediately after kernel launch.
     *
     * @param operation Human-readable operation name.
     */
    inline void throw_if_last_error(const char *operation) {
        throw_if_error(cudaGetLastError(), operation);
    }

    /**
     * @brief Synchronize the current device and throw on failure.
     *
     * Intended for correctness-first algorithm phases where explicit
     * synchronization after kernel launches is useful for deterministic
     * error surfacing and easier debugging.
     *
     * @param operation Human-readable operation name.
     */
    inline void synchronize_or_throw(const char *operation) {
        throw_if_error(cudaDeviceSynchronize(), operation);
    }

    /* ============================================================================
     * Device Buffer
     * ========================================================================== */

    /**
     * @brief RAII wrapper for a contiguous device allocation.
     *
     * Owns a single CUDA device allocation of @p count elements of type @p T.
     * This is intended for algorithm scratch buffers and temporary working
     * memory, not domain objects like prepared CSR graphs.
     *
     * Notes:
     * - @p T is the element type, not the count type
     * - allocation size is tracked in std::size_t
     * - large-graph support is constrained by the graph's index/value types,
     *   not by this wrapper itself
     */
    template<typename T>
    class DeviceBuffer {
    public:
        DeviceBuffer() = default;

        explicit DeviceBuffer(const std::size_t count)
            : count_(count) {
            if (count_ == 0) {
                return;
            }

            throw_if_error(
                cudaMalloc(reinterpret_cast<void **>(&ptr_), count_ * sizeof(T)),
                "cudaMalloc"
            );
        }

        ~DeviceBuffer() {
            reset();
        }

        DeviceBuffer(const DeviceBuffer &) = delete;

        DeviceBuffer &operator=(const DeviceBuffer &) = delete;

        DeviceBuffer(DeviceBuffer &&other) noexcept
            : ptr_(std::exchange(other.ptr_, nullptr)),
              count_(std::exchange(other.count_, 0)) {
        }

        DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
            if (this == &other) {
                return *this;
            }

            reset();
            ptr_ = std::exchange(other.ptr_, nullptr);
            count_ = std::exchange(other.count_, 0);
            return *this;
        }

        [[nodiscard]] T *get() noexcept {
            return ptr_;
        }

        [[nodiscard]] const T *get() const noexcept {
            return ptr_;
        }

        [[nodiscard]] std::size_t size() const noexcept {
            return count_;
        }

    private:
        void reset() noexcept {
            if (ptr_ != nullptr) {
                cudaFree(ptr_);
                ptr_ = nullptr;
            }
            count_ = 0;
        }

    private:
        T *ptr_{nullptr};
        std::size_t count_{0};
    };
} // namespace gpupath::cuda
