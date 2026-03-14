// file: cpp/src/cuda_bootstrap.cu

#include "gpupath/cuda_bootstrap.hpp"

#include <cuda_runtime.h>

#include <sstream>
#include <string>

namespace gpupath {
    namespace {
        /**
         * @brief Probe CUDA device visibility with the smallest safe runtime call.
         *
         * @param device_count_out Receives the number of visible CUDA devices when
         *        the call succeeds. Set to 0 on failure.
         * @return Raw CUDA status code from cudaGetDeviceCount.
         */
        int probe_device_count(int &device_count_out) {
            device_count_out = 0;
            const cudaError_t status = cudaGetDeviceCount(&device_count_out);
            return static_cast<int>(status);
        }
    }  // namespace

    bool cuda_available() {
        int device_count = 0;
        const int status_code = probe_device_count(device_count);
        return status_code == static_cast<int>(cudaSuccess) && device_count > 0;
    }

    std::string cuda_version_string() {
        int device_count = 0;
        const int status_code = probe_device_count(device_count);

        std::ostringstream oss;
        oss << "cuda bootstrap"
            << " status_code=" << status_code
            << " device_count=" << device_count;

        return oss.str();
    }

    std::string cuda_info_json() {
        int device_count = 0;
        const int status_code = probe_device_count(device_count);

        std::ostringstream oss;
        oss << "{"
            << "\"cuda_available\":"
            << ((status_code == static_cast<int>(cudaSuccess) && device_count > 0) ? "true" : "false")
            << ","
            << "\"status_code\":" << status_code
            << ","
            << "\"device_count\":" << device_count
            << "}";

        return oss.str();
    }

}  // namespace gpupath