#pragma once

namespace psyne {

enum class Opcode : uint8_t {
  NONE = 0x00,
  PUT = 0x10,
  GET = 0x11,
};

}
