// file: cpp/include/gpupath/cuda_bootstrap.hpp

#pragma once

#include <string>

namespace gpupath {
    /**
     * @brief Return whether a CUDA-capable device/runtime is available.
     *
     * This performs a lightweight CUDA runtime probe and returns false instead
     * of throwing if the runtime/device is unavailable.
     *
     * @return True if at least one CUDA device is visible, otherwise false.
     */
    bool cuda_available();

    /**
     * @brief Return a human-readable CUDA bootstrap summary string.
     *
     * Intended only for early smoke testing.
     *
     * @return Summary string describing CUDA runtime/device availability.
     */
    std::string cuda_version_string();

    /**
     * @brief Return structured CUDA environment information as a JSON string.
     *
     * The pybind layer can expose this as a Python dict by parsing/assembling
     * fields directly, but keeping the native side simple is useful during
     * bootstrap.
     *
     * @return JSON string describing CUDA runtime/device state.
     */
    std::string cuda_info_json();
} // namespace gpupath
