# Python bindings for Psyne using pybind11
cmake_minimum_required(VERSION 3.16)

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Try to find pybind11
find_package(pybind11 QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, trying to use pip installed version...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE pybind11_RESULT
    )
    
    if(pybind11_RESULT EQUAL 0)
        find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})
        message(STATUS "Found pybind11 via pip: ${pybind11_DIR}")
    else()
        message(FATAL_ERROR "pybind11 not found. Please install with: pip install pybind11")
    endif()
endif()

# Create the Python module
pybind11_add_module(psyne_python psyne_bindings.cpp)

# Set properties
target_compile_features(psyne_python PRIVATE cxx_std_20)

# Link against the main psyne library
target_link_libraries(psyne_python PRIVATE psyne)

# Include directories
target_include_directories(psyne_python PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/utils
    $ENV{HOME}/boost
)

# Set the output name to just "psyne" (without the _python suffix)
set_target_properties(psyne_python PROPERTIES OUTPUT_NAME psyne)

# Compiler-specific options
target_compile_definitions(psyne_python PRIVATE VERSION_INFO="${PROJECT_VERSION}")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(psyne_python PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install the module
install(TARGETS psyne_python DESTINATION .)