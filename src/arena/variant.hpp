#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>

namespace psyne {

// ───────────── fixed 8-byte header ─────────────
struct alignas(8) VariantHdr {
  uint8_t type{0};      // opcode or value kind
  uint8_t rank{0};      // 0 = scalar, 1 = vec, 2+ = tensor
  uint8_t elemBytes{0}; // 1,2,4,8 …
  uint8_t reserved{0};
  uint32_t byteLen{0}; // payload size
};

// ────────────────────────────────────────────────
//   1. Generic templated “lens”  (now with offset)
// ────────────────────────────────────────────────
template <class T, std::size_t Rank> struct VariantView {
  const VariantHdr *hdr{};
  std::size_t byteOffset{0}; // NEW

  const std::byte *raw() const {
    return reinterpret_cast<const std::byte *>(hdr + 1) + byteOffset;
  }
  std::size_t count() const { return (hdr->byteLen - byteOffset) / sizeof(T); }

  // scalar
  T value() const
    requires(Rank == 0)
  {
    static_assert(std::is_trivially_copyable_v<T>);
    T v;
    std::memcpy(&v, raw(), sizeof(T));
    return v;
  }

  // 1-D span
  std::span<const T> span() const
    requires(Rank == 1)
  {
    return {reinterpret_cast<const T *>(raw()), count()};
  }

  // N-D extents stored after header (+ offset)
  std::array<std::size_t, Rank> shape() const
    requires(Rank >= 2)
  {
    const std::size_t *dims = reinterpret_cast<const std::size_t *>(raw());
    std::array<std::size_t, Rank> a{};
    std::memcpy(a.data(), dims, sizeof(std::size_t) * Rank);
    return a;
  }
  const T *data() const
    requires(Rank >= 2)
  {
    return reinterpret_cast<const T *>(raw() + sizeof(std::size_t) * Rank);
  }
};

// ────────────────────────────────────────────────
//   2. Convenience aliases (unchanged)
// ────────────────────────────────────────────────
using Int64ScalarView = VariantView<int64_t, 0>;
using Float32ScalarView = VariantView<float, 0>;
using Float32ArrayView = VariantView<float, 1>;
using Int64ArrayView = VariantView<int64_t, 1>;
template <std::size_t R> using Float32Tensor = VariantView<float, R>;
using JsonView = VariantView<char, 1>;
using EigenVectorView = VariantView<float, 1>;

// ────────────────────────────────────────────────
//   3. Helper – build a lens with optional offset
//      Example:  as<float,1>(hdr, 16)   // skip 16 bytes
// ────────────────────────────────────────────────
template <class T, std::size_t Rank = 0>
VariantView<T, Rank> as(const VariantHdr *h, std::size_t byteOffset = 0) {
  return {h, byteOffset};
}

} // namespace psyne