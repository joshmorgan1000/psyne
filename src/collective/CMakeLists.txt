# Collective operations library
add_library(psyne_collective STATIC
    collective.cpp
)

target_include_directories(psyne_collective
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(psyne_collective
    PUBLIC
        psyne
    PRIVATE
        Threads::Threads
)

# Set properties
set_target_properties(psyne_collective PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)