// file: cpp/src/cuda_bootstrap.cu

#include "gpupath/cuda_bootstrap.hpp"

#include <cuda_runtime.h>

namespace gpupath {
    namespace {
        /**
         * @brief Build a small device-info snapshot for the selected device.
         *
         * @param device_index CUDA device index.
         * @return Optional device info. Empty if querying properties fails.
         */
        std::optional<CudaDeviceInfo> query_device_info(const int device_index) {
            cudaDeviceProp prop{};
            const cudaError_t status = cudaGetDeviceProperties(&prop, device_index);
            if (status != cudaSuccess) {
                return std::nullopt;
            }

            CudaDeviceInfo info;
            info.index = device_index;
            info.name = prop.name;
            info.major = prop.major;
            info.minor = prop.minor;
            info.total_global_memory = static_cast<std::size_t>(prop.totalGlobalMem);
            info.multi_processor_count = prop.multiProcessorCount;
            return info;
        }
    }  // namespace

    CudaBootstrapInfo query_cuda_bootstrap_info() {
        CudaBootstrapInfo info{};

        int device_count = 0;
        const cudaError_t count_status = cudaGetDeviceCount(&device_count);

        info.status_code = static_cast<int>(count_status);
        info.status_name = cudaGetErrorName(count_status);
        info.status_message = cudaGetErrorString(count_status);

        if (count_status != cudaSuccess) {
            return info;
        }

        info.device_count = device_count;
        info.cuda_available = device_count > 0;

        int runtime_version = 0;
        if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
            info.runtime_version = runtime_version;
        }

        int driver_version = 0;
        if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
            info.driver_version = driver_version;
        }

        if (device_count > 0) {
            info.primary_device = query_device_info(0);
        }

        return info;
    }

    bool cuda_available() {
        const CudaBootstrapInfo info = query_cuda_bootstrap_info();
        return info.cuda_available;
    }

}  // namespace gpupath