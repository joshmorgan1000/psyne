#pragma once

#include <cstdint>
#include <type_traits>
#include <typeinfo>
#include <cstring>

namespace psyne {

enum class VariantType : uint8_t {
    None = 0,
    Float32 = 1,
    Float64 = 2,
    Int8 = 3,
    Int16 = 4,
    Int32 = 5,
    Int64 = 6,
    Uint8 = 7,
    Uint16 = 8,
    Uint32 = 9,
    Uint64 = 10,
    Float32Array = 11,
    Float64Array = 12,
    Int8Array = 13,
    Int32Array = 14,
    Custom = 255
};

enum class VariantFlags : uint8_t {
    None = 0,
    GpuBuffer = 1 << 0,
    Readonly = 1 << 1,
    Quantized = 1 << 2,
    Compressed = 1 << 3
};

struct VariantHdr {
    uint8_t type;
    uint8_t flags;
    uint16_t reserved;
    uint32_t byteLen;
    
    void* data() { return reinterpret_cast<uint8_t*>(this) + sizeof(VariantHdr); }
    const void* data() const { return reinterpret_cast<const uint8_t*>(this) + sizeof(VariantHdr); }
    
    template<typename T>
    T* as() { return static_cast<T*>(data()); }
    
    template<typename T>
    const T* as() const { return static_cast<const T*>(data()); }
    
    size_t element_count() const {
        switch (static_cast<VariantType>(type)) {
            case VariantType::Float32:
            case VariantType::Float64:
            case VariantType::Int8:
            case VariantType::Int16:
            case VariantType::Int32:
            case VariantType::Int64:
            case VariantType::Uint8:
            case VariantType::Uint16:
            case VariantType::Uint32:
            case VariantType::Uint64:
                return 1;
            case VariantType::Float32Array:
                return byteLen / sizeof(float);
            case VariantType::Float64Array:
                return byteLen / sizeof(double);
            case VariantType::Int8Array:
                return byteLen / sizeof(int8_t);
            case VariantType::Int32Array:
                return byteLen / sizeof(int32_t);
            default:
                return 0;
        }
    }
};

static_assert(sizeof(VariantHdr) == 8, "VariantHdr must be 8 bytes");
static_assert(alignof(VariantHdr) <= 8, "VariantHdr must be 8-byte aligned");

template<typename T>
struct VariantTraits {
    static constexpr VariantType type() {
        if constexpr (std::is_same_v<T, float>) return VariantType::Float32;
        else if constexpr (std::is_same_v<T, double>) return VariantType::Float64;
        else if constexpr (std::is_same_v<T, int8_t>) return VariantType::Int8;
        else if constexpr (std::is_same_v<T, int16_t>) return VariantType::Int16;
        else if constexpr (std::is_same_v<T, int32_t>) return VariantType::Int32;
        else if constexpr (std::is_same_v<T, int64_t>) return VariantType::Int64;
        else if constexpr (std::is_same_v<T, uint8_t>) return VariantType::Uint8;
        else if constexpr (std::is_same_v<T, uint16_t>) return VariantType::Uint16;
        else if constexpr (std::is_same_v<T, uint32_t>) return VariantType::Uint32;
        else if constexpr (std::is_same_v<T, uint64_t>) return VariantType::Uint64;
        else return VariantType::Custom;
    }
    
    static constexpr VariantType array_type() {
        if constexpr (std::is_same_v<T, float>) return VariantType::Float32Array;
        else if constexpr (std::is_same_v<T, double>) return VariantType::Float64Array;
        else if constexpr (std::is_same_v<T, int8_t>) return VariantType::Int8Array;
        else if constexpr (std::is_same_v<T, int32_t>) return VariantType::Int32Array;
        else return VariantType::Custom;
    }
};

inline VariantFlags operator|(VariantFlags a, VariantFlags b) {
    return static_cast<VariantFlags>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

inline VariantFlags operator&(VariantFlags a, VariantFlags b) {
    return static_cast<VariantFlags>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
}

}  // namespace psyne