#pragma once
#include "arena/variant.hpp"
#include "psyne/opcodes.hpp"
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>

namespace psyne {

/** Simple dispatcher that interprets the opcode stored in the first
 *  VariantHdr of a message slab and invokes the appropriate callback.
 *
 *  This is a minimal helper so the network layer can remain header-only.
 */
class Dispatcher {
public:
  using PutFn = std::function<void(std::span<const float>)>;
  using GetFn = std::function<void(uint64_t)>;

  Dispatcher(PutFn putCb = {}, GetFn getCb = {})
      : onPut_{std::move(putCb)}, onGet_{std::move(getCb)} {}

  void setPutCallback(PutFn fn) { onPut_ = std::move(fn); }
  void setGetCallback(GetFn fn) { onGet_ = std::move(fn); }

  void handleMessage(std::span<const std::byte> slab) const {
    if (slab.size() < sizeof(VariantHdr))
      return;
    auto *hdr = reinterpret_cast<const VariantHdr *>(slab.data());
    auto opcode = static_cast<Opcode>(hdr->type);
    const std::byte *payload = slab.data() + sizeof(VariantHdr);
    switch (opcode) {
    case Opcode::PUT:
      if (onPut_)
        onPut_(std::span<const float>(reinterpret_cast<const float *>(payload),
                                      hdr->byteLen / sizeof(float)));
      break;
    case Opcode::GET:
      if (onGet_ && hdr->byteLen >= sizeof(uint64_t)) {
        uint64_t key;
        std::memcpy(&key, payload, sizeof(key));
        onGet_(key);
      }
      break;
    default:
      break;
    }
  }

private:
  PutFn onPut_;
  GetFn onGet_;
};

} // namespace psyne
