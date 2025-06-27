#pragma once

#include "psyne/opcodes.hpp"
#include "slab.hpp"
#include "variant.hpp"
#include <cstring>
#include <span>
#include <tuple>

namespace psyne {

class VariantBuilder {
public:
  VariantBuilder(Block b) : blk_{b}, cursor_{b.ptr} {}

  template <class T> void addScalar(const T &v) {
    static_assert(std::is_trivially_copyable_v<T>);
    writeHdr(sizeof(T), 0, sizeof(T));
    std::memcpy(cursor_, &v, sizeof(T));
    cursor_ += sizeof(T);
  }

  template <class T>
  void addVector(std::span<const T> v) {
    static_assert(std::is_trivially_copyable_v<T>);
    writeHdr(v.size() * sizeof(T), 1, sizeof(T));
    std::memcpy(cursor_, v.data(), v.size() * sizeof(T));
    cursor_ += v.size() * sizeof(T);
  }

  void addBytes(const void *src, std::size_t n) {
    writeHdr(n, 0, 1); // elemBytes = 1 → raw blob
    std::memcpy(cursor_, src, n);
    cursor_ += n;
  }

  std::span<const std::byte> built() const {
    return {blk_.ptr, static_cast<std::size_t>(cursor_ - blk_.ptr)};
  }

private:
  void writeHdr(std::size_t len, uint8_t rank, uint8_t elemBytes) {
    auto *hdr = reinterpret_cast<VariantHdr *>(cursor_);
    hdr->type = static_cast<uint8_t>(Opcode::NONE);
    hdr->rank = rank;
    hdr->elemBytes = elemBytes;
    hdr->reserved = 0;
    hdr->byteLen = static_cast<uint32_t>(len);
    cursor_ += sizeof(VariantHdr);
  }

  Block blk_;
  std::byte *cursor_;
};

// ─── 1.  Field meta ─────────────────────────────────────────────
template <auto NameLit, class T, std::size_t Rank> struct Field {
  static constexpr auto name = NameLit; // string literal
  using type = T;
  static constexpr std::size_t rank = Rank;
};

// ─── 2.  Schema meta ────────────────────────────────────────────
template <class... Fields> struct Schema {
  static constexpr std::size_t fieldCount = sizeof...(Fields);

  // ---------------- builder ----------------
  class Builder {
  public:
    Builder(Slab &slab) : blk_{slab.alloc(4096)}, vb_{blk_} {}
    template <std::size_t I, class U> void set(const U &value) {
      using F = std::tuple_element_t<I, std::tuple<Fields...>>;
      if constexpr (F::rank == 0)
        vb_.addScalar<U>(value);
      else
        vb_.addVector(std::span<const typename F::type>(value.data(), value.size()));
    }
    Block finish() { return blk_; }

  private:
    Block blk_;
    VariantBuilder vb_;
  };

  // ---------------- view -------------------
  class View {
  public:
    explicit View(const Block &blk) : p_{blk.ptr} {}
    template <std::size_t I> auto get() const {
      using F = std::tuple_element_t<I, std::tuple<Fields...>>;
      VariantView v{p_};
      if constexpr (F::rank == 0)
        return as<typename F::type>(v.hdr).value();
      else
        return as<typename F::type, 1>(v.hdr).span();
    }

  private:
    const VariantHdr *p_;
  };
};

} // namespace psyne