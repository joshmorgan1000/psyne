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

if(DEFINED ENV{GITHUB_ACTIONS} OR "$ENV{CI}" STREQUAL "true")
    message(STATUS "CI detected: disabling tests")
    set(HEAVY_TESTS OFF CACHE BOOL "Disable tests on CI" FORCE)
endif()

# Add tests
add_psyne_test(simple_test)
add_psyne_test(basic_test)
add_psyne_test(message_test)
add_psyne_test(channel_test)
add_psyne_test(compression_encryption_test)
if(HEAVY_TESTS)
    add_psyne_test(simple_memory_test)
    add_psyne_test(performance_test)
    add_psyne_test(tcp_test)
    add_psyne_test(integration_test)
    add_psyne_test(performance_benchmark)
    add_psyne_test(multi_core_benchmark)
endif()