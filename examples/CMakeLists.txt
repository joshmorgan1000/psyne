# Examples CMakeLists.txt

# Basic examples
add_executable(layer_communication basic/layer_communication.cpp)
target_link_libraries(layer_communication PRIVATE Psyne::psyne)

add_executable(ipc_example basic/ipc_example.cpp)
target_link_libraries(ipc_example PRIVATE Psyne::psyne)

add_executable(tcp_example basic/tcp_example.cpp)
target_link_libraries(tcp_example PRIVATE Psyne::psyne)

# Pattern examples
add_executable(gradient_aggregation patterns/gradient_aggregation_simple.cpp)
target_link_libraries(gradient_aggregation PRIVATE Psyne::psyne)

add_executable(data_parallel_broadcast patterns/data_parallel_broadcast.cpp)
target_link_libraries(data_parallel_broadcast PRIVATE Psyne::psyne)

# Templated channel examples
add_executable(templated_spsc_demo templated_spsc_demo.cpp)
target_link_libraries(templated_spsc_demo PRIVATE Psyne::psyne)

add_executable(templated_substrate_demo templated_substrate_demo.cpp)
target_link_libraries(templated_substrate_demo PRIVATE Psyne::psyne)

add_executable(gpu_debug_substrate_demo gpu_debug_substrate_demo.cpp)
target_link_libraries(gpu_debug_substrate_demo PRIVATE Psyne::psyne)

# Add more examples as they are created
# add_executable(gpu_tensor_transfer gpu/gpu_tensor_transfer.cpp)
# target_link_libraries(gpu_tensor_transfer PRIVATE Psyne::psyne)

# Install examples
install(TARGETS layer_communication ipc_example tcp_example gradient_aggregation data_parallel_broadcast templated_spsc_demo templated_substrate_demo gpu_debug_substrate_demo
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/psyne-examples
)