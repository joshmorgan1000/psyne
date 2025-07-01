# Examples CMakeLists.txt

# Get parent variables
set(PSYNE_HAS_BOOST ${Boost_FOUND} PARENT_SCOPE)

# Basic examples (always build)
# TODO: Restore these examples
# add_executable(layer_communication basic/layer_communication.cpp)
# target_link_libraries(layer_communication PRIVATE psyne)

# Pattern examples (always build)
# TODO: Restore these examples
# add_executable(gradient_aggregation patterns/gradient_aggregation_simple.cpp)
# target_link_libraries(gradient_aggregation PRIVATE psyne)

# add_executable(data_parallel_broadcast patterns/data_parallel_broadcast.cpp)
# target_link_libraries(data_parallel_broadcast PRIVATE psyne)

# Network examples (only if Boost found)
if(Boost_FOUND)
    add_executable(tcp_example basic/tcp_example.cpp)
    target_link_libraries(tcp_example PRIVATE psyne)
    
    add_executable(ipc_example basic/ipc_example.cpp)
    target_link_libraries(ipc_example PRIVATE psyne)
    
    # Protocol examples
    add_executable(protocol_demo protocols/protocol_demo.cpp)
    target_link_libraries(protocol_demo PRIVATE psyne)
    
    add_executable(tdt_compression_demo protocols/tdt_compression_demo.cpp)
    target_link_libraries(tdt_compression_demo PRIVATE psyne)
else()
    message(STATUS "Skipping network examples - Boost not found")
endif()

# Concept demonstration examples
# TODO: Restore these examples
# add_executable(concept_based_demo concept_based_demo.cpp)
# target_link_libraries(concept_based_demo PRIVATE psyne)

# add_executable(clean_architecture_demo clean_architecture_demo.cpp)
# target_link_libraries(clean_architecture_demo PRIVATE psyne)

add_executable(behaviors_demo behaviors_demo.cpp)
target_link_libraries(behaviors_demo PRIVATE psyne)

# Debug examples
add_executable(metrics_demo debug/metrics_demo.cpp)
target_link_libraries(metrics_demo PRIVATE psyne)

# GPU examples (if enabled)
if(PSYNE_ENABLE_GPU)
    add_executable(zero_copy_gpu_example gpu/zero_copy_gpu_example.cpp)
    target_link_libraries(zero_copy_gpu_example PRIVATE psyne)
    
    # On macOS, build GPU example as Objective-C++ for Metal support
    if(APPLE AND PSYNE_METAL_ENABLED)
        add_executable(gpu_example basic/gpu_example.mm)
        set_source_files_properties(basic/gpu_example.mm PROPERTIES
            COMPILE_FLAGS "-x objective-c++"
        )
    else()
        add_executable(gpu_example basic/gpu_example.cpp)
    endif()
    
    target_link_libraries(gpu_example PRIVATE psyne)
    
    # Link Metal framework on macOS
    if(APPLE AND PSYNE_METAL_ENABLED)
        target_link_libraries(gpu_example PRIVATE
            "-framework Metal"
            "-framework Foundation"
        )
    endif()
    
    install(TARGETS
        zero_copy_gpu_example
        gpu_example
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/psyne-examples
    )
endif()

# Install examples
install(TARGETS 
    behaviors_demo
    metrics_demo
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/psyne-examples
)

# Install network examples if built
if(Boost_FOUND)
    install(TARGETS 
        tcp_example 
        ipc_example
        protocol_demo
        tdt_compression_demo
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/psyne-examples
    )
endif()