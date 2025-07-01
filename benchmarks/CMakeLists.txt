# Benchmarks CMakeLists.txt

# Add include directories for all benchmarks
include_directories(${CMAKE_SOURCE_DIR}/include/psyne/global)

# Core benchmarks
add_executable(messaging_patterns_benchmark messaging_patterns_benchmark.cpp)
target_link_libraries(messaging_patterns_benchmark PRIVATE psyne)

add_executable(network_saturation_test network_saturation_test.cpp)
target_link_libraries(network_saturation_test PRIVATE psyne)

add_executable(spsc_inprocess_benchmark spsc_inprocess_benchmark.cpp)
target_link_libraries(spsc_inprocess_benchmark PRIVATE psyne)

add_executable(spsc_corrected_benchmark spsc_corrected_benchmark.cpp)
target_link_libraries(spsc_corrected_benchmark PRIVATE psyne)

# Protocol benchmarks (only if Boost found for networking)
if(Boost_FOUND)
    add_executable(tdt_compression_benchmark protocols/tdt_compression_benchmark.cpp)
    target_link_libraries(tdt_compression_benchmark PRIVATE psyne)
    
    # Network-based benchmarks
    add_executable(tcp_tdt_benchmark tcp_tdt_benchmark.cpp)
    target_link_libraries(tcp_tdt_benchmark PRIVATE psyne)
endif()

# Pattern benchmarks
add_executable(all_patterns_test all_patterns_test.cpp)
target_link_libraries(all_patterns_test PRIVATE psyne)

add_executable(beast_mode_test beast_mode_test.cpp)
target_link_libraries(beast_mode_test PRIVATE psyne)

# Comprehensive benchmarks
add_executable(comprehensive_v2_benchmark comprehensive_v2_benchmark.cpp)
target_link_libraries(comprehensive_v2_benchmark PRIVATE psyne)

add_executable(comprehensive_v2_working comprehensive_v2_working.cpp)
target_link_libraries(comprehensive_v2_working PRIVATE psyne)

add_executable(comprehensive_all_substrates comprehensive_all_substrates.cpp)
target_link_libraries(comprehensive_all_substrates PRIVATE psyne)

add_executable(quick_all_substrates quick_all_substrates.cpp)
target_link_libraries(quick_all_substrates PRIVATE psyne)

add_executable(quick_beast_test quick_beast_test.cpp)
target_link_libraries(quick_beast_test PRIVATE psyne)

# Set benchmark-specific compile options for performance
foreach(benchmark IN ITEMS 
    messaging_patterns_benchmark
    network_saturation_test
    spsc_inprocess_benchmark
    spsc_corrected_benchmark
    all_patterns_test
    beast_mode_test
    comprehensive_v2_benchmark
    comprehensive_v2_working
    comprehensive_all_substrates
    quick_all_substrates
    quick_beast_test)
    
    if(TARGET ${benchmark})
        target_compile_options(${benchmark} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native>
            $<$<CXX_COMPILER_ID:MSVC>:/O2>
        )
    endif()
endforeach()

# Also set for network benchmarks if they exist
if(Boost_FOUND)
    foreach(benchmark IN ITEMS tdt_compression_benchmark tcp_tdt_benchmark)
        if(TARGET ${benchmark})
            target_compile_options(${benchmark} PRIVATE
                $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native>
                $<$<CXX_COMPILER_ID:MSVC>:/O2>
            )
        endif()
    endforeach()
endif()

# Install benchmarks
install(TARGETS 
    messaging_patterns_benchmark 
    network_saturation_test
    spsc_inprocess_benchmark
    spsc_corrected_benchmark
    all_patterns_test
    beast_mode_test
    comprehensive_v2_benchmark
    comprehensive_v2_working
    comprehensive_all_substrates
    quick_all_substrates
    quick_beast_test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/psyne-benchmarks
)

# Install network benchmarks if built
if(Boost_FOUND)
    install(TARGETS 
        tdt_compression_benchmark
        tcp_tdt_benchmark
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/psyne-benchmarks
    )
endif()

# Copy benchmark script
install(PROGRAMS run_benchmarks.sh 
    DESTINATION ${CMAKE_INSTALL_BINDIR}/psyne-benchmarks
)