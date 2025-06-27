# Psyne — High‑level Architecture

> **Goal**  A zero‑copy, event‑driven message pipeline that can run the *same* binary payload through CPU logic **and** a GPU compute kernel.

---

## 1. End‑to‑End Data Flow

```
┌──────────────┐   TCP   ┌─────────────┐   dispatch   ┌──────────────┐
│  Sender App  │──────▶  │  Listener   │─────────────▶│  GPU Ring /  │
│  (build slab)│         │ (Asio + IO) │              │ Vulkan Pipe  │
└──────────────┘         └─────────────┘              └──────────────┘
        ▲                        │                          ▲
        │                        │ host_ptr (shared)        │ results land in
        │                        ▼                          │ output buffer
        └─────── same byte‑array (slab) ────────────────────┘
```

* **Sender** writes a contiguous *slab* in one `std::vector<std::byte>`.
* **Listener** (Boost.Asio coroutine) reads `<len><payload>`; no blocking threads.
* **Dispatcher** interprets the first `VariantHdr.type` as an **opcode**.
* **GPU path** copies the *same* slab into the current `ShaderBufferRing` slot and kicks a compute dispatch.

---

## 2. Slab Wire Format

| Offset | Bytes | Field               | Notes                      |
| -----: | ----: | ------------------- | -------------------------- |
|      0 |     4 | **len**             | Total bytes after this u32 |
|      4 |  (32) | *hash* *(future)*   | BLAKE3 of payload          |
|   4+32 |     8 | **VariantHdr** (1)  | opcode lives in `type`     |
|      … |     n | payload for hdr (1) | 8‑byte aligned             |
|      … |     8 | VariantHdr (2)      | optional extras            |
|      … |     m | payload (2)         |                            |

### `VariantHdr` (8 bytes, 8‑byte aligned)

```cpp
struct VariantHdr {
    uint8_t  type;    // opcode or value kind
    uint8_t  flags;   // bit0=isArray, bits1‑3=log2(elemSize)
    uint16_t reserved;
    uint32_t byteLen; // payload size in bytes
};
```

---

## 3. Memory‑View API (header‑only)

```cpp
using ViewVariant = std::variant<VariantView, Int64View, StrView /* … */>;

ViewVariant viewAs(const VariantHdr* hdr);
```

* **`VariantView`** – base interface (`data()`, `bytes()`).
* **`Int64View`, `StrView`, `SpanView<T>`** – CRTP helpers for typed access.
* `viewAs()` returns a `std::variant` so caller can `std::visit()`.

---

## 4. Network Layer (header‑only)

```cpp
asio::awaitable<void> listener(uint16_t port);
```

* Uses **Boost.Asio** coroutines (`co_await`).
* Length‑prefixed reads; resizes one reusable `std::vector<std::byte>`.
* Calls `psyne::handleMessage(std::span<const std::byte>)`.

---

## 5. Dispatcher Stub

```cpp
void handleMessage(std::span<const std::byte> slab) {
    const VariantHdr* h = reinterpret_cast<const VariantHdr*>(slab.data());
    auto v = viewAs(h);
    std::visit([](auto&& vw){ /* opcode / type switch */ }, v);
}
```

* First `VariantHdr.type` doubles as **opcode** (e.g., `OP_PUT = 0x10`).
* Later variants in the same slab can carry parameters or payloads.

---

## 6. GPU Integration

* One `ShaderBufferRing` owns N pairs *(input, output)* of host‑visible Vulkan buffers.
* After `handleMessage`, copy slab bytes into the **current** input buffer.
* Record a compute dispatch; output lands in the matching output buffer.
* CPU polls or fences, then interprets results with another `viewAs()` pass.

---

## 7. Extension Points

1. **BLAKE3 digest** for integrity – insert 32 bytes after length, verify before dispatch.
2. **DSL Byte‑code** opcode – compile once, cache by hash, run via LLVM ORC JIT or GPU interpreter.
3. **TLS** – wrap listener’s socket in `asio::ssl::stream` without touching the slab.
4. **Multi‑backend GPU** – keep the same slab; drop in Metal‑cpp buffer allocation on macOS.

---

## 8. Build‑time Summary

* Header‑only for Variant, Dispatcher, Listener.
* Only `.cpp` you must compile today: the main that spawns `listener()`, plus GPU/Vulkan helpers.
* No `DYLD_LIBRARY_PATH` needed: link `-Wl,-rpath,${VULKAN_SDK}/lib` or use static MoltenVK.
