#include "psyne/channel/tcp_protocol.hpp"
#include "../src/utils/xxhash32.h"

namespace psyne {

uint32_t calculate_xxhash32(const void* data, size_t len) {
    return XXHash32::hash(data, len, 0);
}

}  // namespace psyne