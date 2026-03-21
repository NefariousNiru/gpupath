cmake_minimum_required(VERSION 3.18)
project(gpupath LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Add APIs here
pybind11_add_module(_native
        cpp/module.cpp
        cpp/src/bfs.cpp
        cpp/src/sssp.cpp
        cpp/src/native_csr_graph.cpp
        cpp/src/multi_source_lengths.cpp
        cpp/src/cuda_bootstrap.cu
)

# Compile to package _native used by Python to import
target_include_directories(_native PRIVATE cpp/include)
target_compile_features(_native PRIVATE cxx_std_17)
set_target_properties(_native PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 75
)

# Non Exposed tests and Target
add_executable(cuda_bootstrap_smoke
        cpp/tests/cuda_bootstrap_smoke.cpp
        cpp/src/cuda_bootstrap.cu
)
target_include_directories(cuda_bootstrap_smoke PRIVATE cpp/include)
target_compile_features(cuda_bootstrap_smoke PRIVATE cxx_std_17)
set_target_properties(cuda_bootstrap_smoke PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 75
)

# Install all Targets
install(TARGETS _native DESTINATION gpupath)