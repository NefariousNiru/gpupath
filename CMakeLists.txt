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

# Default ON for local development.
# Configure with -DBUILD_TESTING=OFF to skip building/registering tests for prod.
option(BUILD_TESTING "Build gpupath native tests" ON)

# Expose native implementation objects here; do not include module.cpp.
add_library(gpupath_native_objs OBJECT
        cpp/src/bfs.cpp
        cpp/src/sssp.cpp
        cpp/src/native_csr_graph.cpp
        cpp/src/multi_source_lengths.cpp
        cpp/src/cuda_bootstrap.cu
        cpp/src/cuda_csr_graph.cu
)

# Compile to package gpupath_native_objs used by pybind
target_include_directories(gpupath_native_objs PRIVATE cpp/include)
target_compile_features(gpupath_native_objs PRIVATE cxx_std_17)
set_target_properties(gpupath_native_objs PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 75
)

# Bind to Python with pybind
pybind11_add_module(_native
        cpp/module.cpp
        $<TARGET_OBJECTS:gpupath_native_objs>
)

# Compile to package _native used by Python to import
target_include_directories(_native PRIVATE cpp/include)
target_compile_features(_native PRIVATE cxx_std_17)
set_target_properties(_native PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 75
)

# Tests live under cpp/tests/CMakeLists.txt and use add_test(...).
if (BUILD_TESTING)
    enable_testing()
    add_subdirectory(cpp/tests)
endif ()

# Install all Targets
install(TARGETS _native DESTINATION gpupath)