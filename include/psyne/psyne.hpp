#pragma once

// Core components
#include "core/variant.hpp"
#include "core/variant_view.hpp"
#include "core/message.hpp"
#include "core/message_builder.hpp"
#include "core/fixed_message.hpp"

// Memory management
#include "memory/slab_allocator.hpp"
#include "memory/ring_buffer.hpp"

// Channels
#include "channel/channel.hpp"
#include "channel/ipc_channel.hpp"

// Version information
#define PSYNE_VERSION_MAJOR 0
#define PSYNE_VERSION_MINOR 1
#define PSYNE_VERSION_PATCH 0

// Common standard library includes that users will need
#include <chrono>
#include <thread>
#include <memory>
#include <optional>
#include <functional>
#include <iostream>

// Platform-specific includes
#ifdef __APPLE__
    #include <sys/mman.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <semaphore.h>
#endif

// Main namespace
namespace psyne {
    // Version string
    constexpr const char* version() {
        return "0.1.0";
    }
}