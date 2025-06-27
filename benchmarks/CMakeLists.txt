# Psyne Benchmarks
cmake_minimum_required(VERSION 3.20)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    $ENV{HOME}/boost
)

# Function to add a benchmark executable
function(add_psyne_benchmark name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} 
        psyne
        Threads::Threads
        Eigen3::Eigen
    )
    target_compile_features(${name} PRIVATE cxx_std_20)
    
    # Release mode optimizations
    target_compile_options(${name} PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-ffast-math>
    )
    
    # Add compiler warnings
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

# Add benchmarks
add_psyne_benchmark(throughput_benchmark)
add_psyne_benchmark(latency_benchmark)
add_psyne_benchmark(dynamic_allocation_benchmark)

# Create a custom target to run all benchmarks
add_custom_target(run_benchmarks
    COMMAND throughput_benchmark
    COMMAND latency_benchmark  
    COMMAND dynamic_allocation_benchmark
    DEPENDS throughput_benchmark latency_benchmark dynamic_allocation_benchmark
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running all benchmarks..."
)

# Create a custom target to run benchmarks and save results
add_custom_target(benchmark_report
    COMMAND ${CMAKE_COMMAND} -E echo "Running benchmarks and generating report..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/results
    COMMAND throughput_benchmark > ${CMAKE_CURRENT_BINARY_DIR}/results/throughput_results.txt
    COMMAND latency_benchmark > ${CMAKE_CURRENT_BINARY_DIR}/results/latency_results.txt
    COMMAND dynamic_allocation_benchmark > ${CMAKE_CURRENT_BINARY_DIR}/results/dynamic_results.txt
    COMMAND ${CMAKE_COMMAND} -E echo "Results saved to ${CMAKE_CURRENT_BINARY_DIR}/results/"
    DEPENDS throughput_benchmark latency_benchmark dynamic_allocation_benchmark
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating benchmark report..."
)