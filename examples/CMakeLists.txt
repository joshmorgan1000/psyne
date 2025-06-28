set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-fsanitize=address -fno-omit-frame-pointer" CACHE STRING "C++ compiler flags")
set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "Minimum macOS version")
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/psyne
    ${CMAKE_SOURCE_DIR}/src/utils
    ${EIGEN3_INCLUDE_DIR}
    ${Vulkan_INCLUDE_DIRS}
    $ENV{HOME}/boost
)
# ----------------------------------------------------------------------------------------
# Examples
# ----------------------------------------------------------------------------------------

# Find required packages
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

# Function to add an example executable
function(add_psyne_example name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} 
        psyne
        Threads::Threads
        Eigen3::Eigen
    )
    target_compile_features(${name} PRIVATE cxx_std_20)
    
    # Add compiler warnings
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

# Add examples
add_psyne_example(simple_messaging)
add_psyne_example(producer_consumer)
add_psyne_example(multi_type_channel)
add_psyne_example(fixed_size_demo)
add_psyne_example(coroutine_example)
add_psyne_example(channel_factory_demo)
add_psyne_example(dynamic_allocation_demo)
add_psyne_example(ipc_producer)
add_psyne_example(ipc_consumer)
add_psyne_example(ipc_test)
add_psyne_example(tcp_server)
add_psyne_example(tcp_client)
add_psyne_example(unix_socket_demo)
add_psyne_example(enhanced_types_demo)
add_psyne_example(performance_demo)
add_psyne_example(high_performance_messaging)
add_psyne_example(metrics_demo)
add_psyne_example(compression_demo)
add_psyne_example(debug_demo)
add_psyne_example(udp_multicast_demo)
add_psyne_example(simple_enhanced_types_test)
add_psyne_example(rdma_demo)
add_psyne_example(debug_multicast)
add_psyne_example(routing_demo_final)
add_psyne_example(test_bytevector)
add_psyne_example(websocket_demo)

# GPU examples (optional)
if(PSYNE_GPU_SUPPORT)
    add_psyne_example(gpu_vector_demo)
endif()

# ----------------------------------------------------------------------------------------
# Tests (commented out for now)
# ----------------------------------------------------------------------------------------
# add_executable(psyne_basic_test psyne_tests.cpp)
# target_link_libraries(psyne_basic_test PRIVATE psyne)

# add_executable(psyne_rpc_stress rpc_stress.cpp)
# target_link_libraries(psyne_rpc_stress PRIVATE psyne)