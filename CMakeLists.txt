cmake_minimum_required(VERSION 3.18)
project(gpupath LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_native
        cpp/src/module.cpp
        cpp/src/bfs.cpp
)

target_include_directories(_native PRIVATE cpp/include)
target_compile_features(_native PRIVATE cxx_std_17)

install(TARGETS _native DESTINATION gpupath)