#pragma once

/**
 * @file psyne.hpp
 * @brief Main header file for Psyne v2.0 messaging library
 *
 * This header includes all the core components needed for zero-copy
 * message passing with the new substrate + pattern + message architecture.
 */

// Core v2.0 architecture
#include "concepts/substrate_concepts.hpp"
#include "core/behaviors.hpp"

// Substrates (physical layer) - Note: these may have include issues, use
// behaviors.hpp instead #include "channel/substrate/in_process.hpp" #include
// "channel/substrate/ipc.hpp" #include "channel/substrate/tcp_simple.hpp"

// Patterns (coordination layer) - Note: these have logger.hpp dependencies
// #include "channel/pattern/spsc.hpp"
// #include "channel/pattern/mpsc.hpp"
// #include "channel/pattern/spmc.hpp"
// #include "channel/pattern/mpmc.hpp"

// Message types
#include "message/numeric_types.hpp"
#include "message/substrate_aware_types.hpp"

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