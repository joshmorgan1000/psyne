# GPU support module
set(GPU_SOURCES
    gpu_context.cpp
    gpu_message.cpp
)

set(GPU_HEADERS
    ${CMAKE_SOURCE_DIR}/include/psyne/gpu/gpu_buffer.hpp
    ${CMAKE_SOURCE_DIR}/include/psyne/gpu/gpu_message.hpp
)

# Platform-specific GPU implementations
if(APPLE)
    # Metal support
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALKIT_FRAMEWORK MetalKit)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    
    if(METAL_FRAMEWORK)
        message(STATUS "Found Metal framework")
        
        list(APPEND GPU_SOURCES
            metal/metal_buffer.cpp
        )
        
        list(APPEND GPU_HEADERS
            ${CMAKE_SOURCE_DIR}/include/psyne/gpu/metal/metal_buffer.hpp
        )
        
        # Metal shaders
        set(METAL_SHADERS
            metal/shaders/compute_kernels.metal
        )
        
        # Compile Metal shaders
        foreach(shader ${METAL_SHADERS})
            get_filename_component(shader_name ${shader} NAME_WE)
            add_custom_command(
                OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${shader_name}.metallib
                COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/${shader} -o ${CMAKE_CURRENT_BINARY_DIR}/${shader_name}.air
                COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/${shader_name}.air -o ${CMAKE_CURRENT_BINARY_DIR}/${shader_name}.metallib
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${shader}
                COMMENT "Compiling Metal shader ${shader}"
            )
            list(APPEND COMPILED_SHADERS ${CMAKE_CURRENT_BINARY_DIR}/${shader_name}.metallib)
        endforeach()
        
        add_custom_target(metal_shaders ALL DEPENDS ${COMPILED_SHADERS})
        
        set(GPU_ENABLED TRUE PARENT_SCOPE)
        set(GPU_BACKEND "Metal" PARENT_SCOPE)
    endif()
endif()

if(GPU_ENABLED)
    # Create GPU library
    add_library(psyne_gpu STATIC ${GPU_SOURCES} ${GPU_HEADERS})
    
    target_include_directories(psyne_gpu
        PUBLIC
            ${CMAKE_SOURCE_DIR}/include
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    target_link_libraries(psyne_gpu
        PUBLIC
            psyne
        PRIVATE
            Eigen3::Eigen
    )
    
    if(APPLE AND METAL_FRAMEWORK)
        target_link_libraries(psyne_gpu
            PRIVATE
                ${METAL_FRAMEWORK}
                ${FOUNDATION_FRAMEWORK}
        )
        
        target_compile_definitions(psyne_gpu
            PUBLIC
                PSYNE_GPU_SUPPORT
                PSYNE_METAL_SUPPORT
        )
        
        # Install Metal shaders
        install(FILES ${COMPILED_SHADERS}
                DESTINATION share/psyne/shaders)
        
        # Add dependency on shader compilation
        add_dependencies(psyne_gpu metal_shaders)
    endif()
    
    # Set properties
    set_target_properties(psyne_gpu PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    
    message(STATUS "GPU support enabled: ${GPU_BACKEND}")
else()
    message(STATUS "GPU support disabled")
endif()