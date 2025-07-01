# Tests for Psyne v2.0

# Find required packages
find_package(Threads REQUIRED)

# Helper function to add a test
function(add_psyne_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(${test_name} PRIVATE Threads::Threads)
    target_compile_features(${test_name} PRIVATE cxx_std_20)
    
    # Add to CTest
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 30)
endfunction()

# Add all test executables
add_psyne_test(test_spsc_basic)
add_psyne_test(test_all_patterns)
add_psyne_test(test_ipc_substrate)
add_psyne_test(test_tcp_working)
add_psyne_test(test_mpmc_simple)

# Platform-specific settings
if(UNIX AND NOT APPLE)
    # Linux-specific flags
    foreach(test_target test_spsc_basic test_all_patterns test_ipc_substrate test_tcp_working test_mpmc_simple)
        target_link_libraries(${test_target} PRIVATE rt)
    endforeach()
endif()

# Enable testing
enable_testing()