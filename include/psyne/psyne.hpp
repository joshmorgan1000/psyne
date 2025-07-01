#pragma once

/**
 * @file psyne.hpp
 * @brief Main header file for the Psyne messaging library
 *
 * This header includes all the core components needed for zero-copy
 * message passing in AI/ML applications.
 */

// Core channel functionality
#include "psyne/channel/channel.hpp"
#include "psyne/channel/channel_base.hpp"
#include "psyne/channel/spsc_channel.hpp"

// Memory management
#include "psyne/memory/memory_slab.hpp"

// Message types
#include "psyne/core/tensor_message.hpp"

// Version information
#define PSYNE_VERSION_MAJOR 1
#define PSYNE_VERSION_MINOR 3
#define PSYNE_VERSION_PATCH 0
#define PSYNE_VERSION_STRING "1.3.0"

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