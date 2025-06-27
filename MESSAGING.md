# Psyne — High‑level Architecture

> **Goal**  A zero‑copy, event‑driven message pipeline that can run the *same* binary payload through CPU logic **and** a GPU compute kernel.

---

## 2. Messaging Layer (MVP)

### Wire format

* **4 bytes** `len` – little‑endian payload bytes that follow.
* **20 bytes** `keyHash` – BLAKE3‑keyed HMAC (also the consistent‑hash key).
* **1 byte** `op` – `0 = PUT`, `1 = GET` (extend later).
* **4 bytes** `payloadLen` (only for PUT).
* **N bytes** `ciphertext` – encrypted value (PUT) or empty (GET).

### Flow

```
   client ▸ psyne::Socket::sendPut(key, ciphertext)
          ▸  [wire‑encode]  ▸  tcp:5010
                               │
                               ▼
                    𝗕𝗿𝗼𝗸𝗲𝗿  (Router session)
                    ────────────────
                    • verify len only
                    • extract 20‑byte keyHash
                    • shard = ringHash(keyHash)
                    • enqueue Msg into shardQ[shard]
                               │
                               ▼
               𝗦𝗵𝗮𝗿𝗱 thread (Worker)
               ──────────────────────
               • decrypt if PUT
               • table.put / get
               • (future) respond via same socket
```

### Header‑only API sketch

```cpp
// psyne/net/Socket.hpp (header‑only)
class Socket {
    asio::ip::tcp::socket sock;  // owned
public:
    Socket(asio::ip::tcp::socket s) : sock(std::move(s)) {}

    awaitable<void> sendPut(std::span<const uint8_t,20> key,
                            std::span<const uint8_t> cipher) {
        uint32_t len = 20 + 1 + 4 + cipher.size();
        std::vector<uint8_t> buf(4 + len);
        std::memcpy(buf.data(), &len, 4);
        std::memcpy(buf.data()+4, key.data(), 20);
        uint8_t* p = buf.data()+4+20;
        *p++ = 0; // op PUT
        uint32_t pay = cipher.size();
        std::memcpy(p,&pay,4); p+=4;
        std::memcpy(p, cipher.data(), cipher.size());
        co_await asio::async_write(sock, asio::buffer(buf), asio::use_awaitable);
    }
};
```

### Memory handling

* Broker **does not copy** payload: the `vector<uint8_t>` from the session becomes the slab chunk pointer passed to the shard thread.
* ShardWorker stores the pointer or copies into its own arena then calls `slabPool.free(ptr)` when done.

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

---

> **Next Step**  Implement `VariantBuilder` so sender code isn’t raw `memcpy`, then add `OP_PUT` & `OP_GET` that push tensors into the GPU ring.
