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

# Add examples - All C++ files in examples directory
# Core messaging examples
add_psyne_example(simple_messaging)
add_psyne_example(simple_messaging_zero_copy)
add_psyne_example(producer_consumer)
add_psyne_example(multi_type_channel)
add_psyne_example(zero_copy_showcase)

# Channel types and patterns
add_psyne_example(channel_factory_demo)
add_psyne_example(channel_patterns_showcase)
add_psyne_example(fixed_size_demo)
add_psyne_example(dynamic_allocation_demo)

# IPC examples
add_psyne_example(ipc_demo)
add_psyne_example(ipc_producer)
add_psyne_example(ipc_consumer)
add_psyne_example(ipc_test)

# Network examples
add_psyne_example(tcp_demo)
add_psyne_example(tcp_server)
add_psyne_example(tcp_client)
add_psyne_example(unix_socket_demo)
add_psyne_example(udp_multicast_demo)
add_psyne_example(websocket_demo)

# WebRTC examples
add_psyne_example(webrtc_demo)
add_psyne_example(webrtc_simple_example)
add_psyne_example(webrtc_p2p_demo)

# Transport protocols
add_psyne_example(quic_demo)
add_psyne_example(rudp_demo)

# Messaging patterns
add_psyne_example(publish_subscribe_demo)
add_psyne_example(request_reply_demo)
add_psyne_example(routing_demo_final)

# Advanced features
add_psyne_example(async_messaging_demo)
add_psyne_example(coroutine_example)
add_psyne_example(modern_cpp20_demo)

# Type system examples
add_psyne_example(enhanced_types_demo)
add_psyne_example(message_types_demo)
add_psyne_example(simple_enhanced_types_test)
add_psyne_example(test_bytevector)
add_psyne_example(test_floatvector)

# Performance and optimization
add_psyne_example(performance_demo) 
add_psyne_example(high_performance_messaging)
add_psyne_example(metrics_demo)
add_psyne_example(simd_demo)
add_psyne_example(custom_allocator_demo)
add_psyne_example(tensor_optimization_demo)

# Compression and debugging
add_psyne_example(compression_demo)
add_psyne_example(debug_demo)
add_psyne_example(debug_multicast)

# Collective operations
add_psyne_example(collective_demo)
add_psyne_example(collective_simple_test)

# Platform-specific
add_psyne_example(windows_test)

# Integration examples
add_psyne_example(arrow_demo)
add_psyne_example(grpc_demo)

# C API example (requires special handling for .c file)
add_executable(c_api_demo c_api_demo.c)
target_link_libraries(c_api_demo 
    psyne
    Threads::Threads
)
target_compile_features(c_api_demo PRIVATE c_std_11)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(c_api_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

# GPU examples (optional)
if(PSYNE_ENABLE_GPU)
    add_psyne_example(gpu_vector_demo)
    
    # CUDA-specific examples
    if(PSYNE_CUDA_ENABLED)
        add_psyne_example(cuda_vector_demo)
        add_psyne_example(test_cuda)
    endif()
    
    # Metal-specific examples (macOS)
    if(PSYNE_METAL_ENABLED)
        add_psyne_example(metal_simple_demo)
    endif()
    
    # Vulkan-specific examples
    if(PSYNE_VULKAN_ENABLED)
        add_psyne_example(vulkan_gpu_demo)
    endif()
endif()
