# Benchmarks CMakeLists.txt

# Performance benchmarks
add_executable(messaging_patterns_benchmark messaging_patterns_benchmark.cpp)
target_link_libraries(messaging_patterns_benchmark PRIVATE Psyne::psyne)

add_executable(network_saturation_test network_saturation_test.cpp)
target_link_libraries(network_saturation_test PRIVATE Psyne::psyne)

# Original benchmarks (if they exist)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/performance_benchmark.cpp)
    add_executable(performance_benchmark performance_benchmark.cpp)
    target_link_libraries(performance_benchmark PRIVATE Psyne::psyne)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/throughput_comparison.cpp)
    add_executable(throughput_comparison throughput_comparison.cpp)
    target_link_libraries(throughput_comparison PRIVATE Psyne::psyne)
endif()

# Install benchmarks
install(TARGETS messaging_patterns_benchmark network_saturation_test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/psyne-benchmarks
)