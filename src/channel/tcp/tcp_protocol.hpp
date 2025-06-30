#pragma once

// TCP protocol implementation
// Updated to use 64-bit checksums for better collision resistance

#include "../../utils/xxhash64.h"
#include <cstddef>
#include <cstdint>

namespace psyne {

// 64-bit hash for improved message validation
inline uint64_t calculate_xxhash64(const void *data, size_t len) {
    return XXHash64::hash(data, len, 0);
}

} // namespace psyne