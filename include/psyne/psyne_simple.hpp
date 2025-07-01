/**
 * @file psyne_simple.hpp  
 * @brief Simplified header for examples without external dependencies
 */

#pragma once

// Core v2.0 architecture - this is all you need!
#include "core/behaviors.hpp"

// Version information
#define PSYNE_VERSION_MAJOR 2
#define PSYNE_VERSION_MINOR 0
#define PSYNE_VERSION_PATCH 0
#define PSYNE_VERSION_STRING "2.0.0-rc"

namespace psyne {

/**
 * @brief Get Psyne version string
 */
inline const char *version() {
    return PSYNE_VERSION_STRING;
}

/**
 * @brief Check if Psyne was built with GPU support
 */
inline bool has_gpu_support() {
#ifdef PSYNE_ENABLE_GPU
    return true;
#else
    return false;
#endif
}

/**
 * @brief Check if CUDA is available
 */
inline bool has_cuda() {
#ifdef PSYNE_CUDA_ENABLED
    return true;
#else
    return false;
#endif
}

} // namespace psyne