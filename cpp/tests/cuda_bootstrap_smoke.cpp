// file: tests/cuda_bootstrap_smoke.cpp

#include "gpupath/cuda_bootstrap.hpp"

#include <iostream>

int main() {
    const auto [cuda_available, status_code, status_name, status_message, device_count, runtime_version, driver_version, primary_device] = gpupath::query_cuda_bootstrap_info();

    std::cout << "cuda_available: " << (cuda_available ? "true" : "false") << '\n';
    std::cout << "status_code: " << status_code << '\n';
    std::cout << "status_name: " << status_name << '\n';
    std::cout << "status_message: " << status_message << '\n';
    std::cout << "device_count: " << device_count << '\n';
    std::cout << "runtime_version: " << runtime_version << '\n';
    std::cout << "driver_version: " << driver_version << '\n';

    if (primary_device.has_value()) {
        const auto &[index, name, major, minor, total_global_memory, multi_processor_count] = *primary_device;
        std::cout << "primary_device.index: " << index << '\n';
        std::cout << "primary_device.name: " << name << '\n';
        std::cout << "primary_device.compute_capability: "
                  << major << "." << minor << '\n';
        std::cout << "primary_device.total_global_memory: "
                  << total_global_memory << '\n';
        std::cout << "primary_device.multi_processor_count: "
                  << multi_processor_count << '\n';
    } else {
        std::cout << "primary_device: none\n";
    }

    if (status_code < 0) {
        std::cerr << "Invalid CUDA bootstrap status.\n";
        return 1;
    }

    return 0;
}