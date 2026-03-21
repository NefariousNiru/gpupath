# Function to abstract test loading
function(add_gpupath_smoke_test target_name test_source)
    add_executable(${target_name}
            ${test_source}
            $<TARGET_OBJECTS:gpupath_native_objs>
    )

    target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/cpp/include)
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    set_target_properties(${target_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES 75
    )

    add_test(NAME ${target_name} COMMAND ${target_name})
endfunction()


# Add Tests here
add_gpupath_smoke_test(cuda_bootstrap_smoke ${CMAKE_CURRENT_SOURCE_DIR}/cuda_bootstrap_smoke.cpp)
add_gpupath_smoke_test(cuda_csr_graph_smoke ${CMAKE_CURRENT_SOURCE_DIR}/cuda_csr_graph_smoke.cpp)