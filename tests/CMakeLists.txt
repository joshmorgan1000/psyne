set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-fsanitize=address -fno-omit-frame-pointer" CACHE STRING "C++ compiler flags")
set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "Minimum macOS version")
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/gpu
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/arena
    ${CMAKE_SOURCE_DIR}/src/encryption
    ${CMAKE_SOURCE_DIR}/src/cortex
    ${CMAKE_SOURCE_DIR}/third_party/BLAKE3/c
    ${Boost_INCLUDE_DIRS}
)
# ----------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------
add_executable(psyne_basic_test psyne_tests.cpp)
target_link_libraries(psyne_basic_test PRIVATE psyne)

add_executable(psyne_rpc_stress rpc_stress.cpp)
target_link_libraries(psyne_rpc_stress PRIVATE psyne)