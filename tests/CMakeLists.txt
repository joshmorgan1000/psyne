set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/psyne
    ${CMAKE_SOURCE_DIR}/src/utils
    ${EIGEN3_INCLUDE_DIR}
    ${Vulkan_INCLUDE_DIRS}
    $ENV{HOME}/boost
)

# Find required packages
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

# Function to add a test executable
function(add_psyne_test name)
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
    
    # Add as a test
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# Add tests
add_psyne_test(simple_test)
add_psyne_test(basic_test)
# add_psyne_test(tcp_test)  # Disabled: requires TCPChannel implementation
add_psyne_test(message_test)
add_psyne_test(channel_test)
# add_psyne_test(performance_test)  # Disabled: requires PerformanceManager and ReliabilityManager implementation
add_psyne_test(compression_encryption_test)
add_psyne_test(integration_test)