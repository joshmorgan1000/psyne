#pragma once

/**
 * @file checksum.hpp
 * @brief Common checksum utilities for Psyne channels
 */

#include "xxhash64.h"
#include <cstddef>
#include <cstdint>

namespace psyne {
namespace utils {

/**
 * @brief Calculate XXH64 checksum with specified seed
 * @param data Data to checksum
 * @param size Size of data in bytes
 * @param seed Hash seed (different seeds for different protocols)
 * @return 64-bit checksum
 */
inline uint64_t calculate_checksum(const void *data, size_t size,
                                   uint64_t seed = 0) {
    return XXHash64::hash(data, size, seed);
}

// Protocol-specific checksum functions with appropriate seeds
namespace tcp {
inline uint64_t calculate_checksum(const void *data, size_t size) {
    return utils::calculate_checksum(data, size, 0);
}
} // namespace tcp

namespace udp {
inline uint64_t calculate_checksum(const void *data, size_t size) {
    return utils::calculate_checksum(data, size, 0);
}
} // namespace udp

namespace unix_socket {
inline uint64_t calculate_checksum(const void *data, size_t size) {
    return utils::calculate_checksum(data, size, 0x12345678);
}
} // namespace unix_socket

} // namespace utils
} // namespace psyne