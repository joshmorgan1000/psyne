// TCP protocol implementation
// TODO: This file needs to be updated when TCP support is added to the new API

#include <cstddef>
#include <cstdint>
#include "../src/utils/xxhash32.h"

namespace psyne {

// Currently unused - will be needed when TCP channels are implemented
uint32_t calculate_xxhash32(const void* data, size_t len) {
    return XXHash32::hash(data, len, 0);
}

}  // namespace psyne