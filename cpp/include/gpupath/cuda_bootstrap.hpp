// file: cpp/include/gpupath/cuda_bootstrap.hpp

#pragma once

#include <optional>
#include <string>

namespace gpupath {
    /**
     * @brief Basic properties for one CUDA device.
     */
    struct CudaDeviceInfo {
        int index = -1;
        std::string name;
        int major = 0;
        int minor = 0;
        std::size_t total_global_memory = 0;
        int multi_processor_count = 0;
    };

    /**
     * @brief Result of a lightweight CUDA runtime/bootstrap probe.
     *
     * This is intentionally small. It is meant for environment validation,
     * not full hardware introspection.
     */
    struct CudaBootstrapInfo {
        bool cuda_available = false;
        int status_code = -1;
        std::string status_name;
        std::string status_message;
        int device_count = 0;
        int runtime_version = 0;
        int driver_version = 0;
        std::optional<CudaDeviceInfo> primary_device;
    };

    /**
     * @brief Query a lightweight snapshot of CUDA runtime/device availability.
     *
     * This function is the canonical bootstrap probe for the native backend.
     * It should remain lightweight and robust even on partially configured
     * systems.
     *
     * @return Structured CUDA bootstrap information.
     */
    CudaBootstrapInfo query_cuda_bootstrap_info();

    /**
     * @brief Convenience helper indicating whether CUDA is usable.
     *
     * @return True if the runtime probe succeeded and at least one device is
     *         visible, otherwise false.
     */
    bool cuda_available();
} // namespace gpupath
