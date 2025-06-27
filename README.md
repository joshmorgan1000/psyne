# Psyne — High‑level Architecture

> **Goal**  A zero‑copy, event‑driven message pipeline that can run the *same* binary payload through CPU logic **and** a GPU compute kernel. It intends to avoid memcpy except when it is absolutely necessary, such as when sending data over the network or between processes.

Target system is currently Apple Silicon, with x86_64 linux in the future. In theory, since the memory is unified, the same memory should be able to be used for:

- std::vector<float>, std::span<float>, float*, etc.
- Eigen3::VectorXf, Eigen3::MatrixXf, etc.
- Vulkan SSBOs, Metal buffers, CUDA for linux, etc.

Quantized types (e.g., `int8_t`, `uint8_t`) should also be supported.

---

## 1. End‑to‑End Data Flow

Each message channel is a *slab* of pre-allocated memory, essentially a ring buffer with a bump pointer.

* A `channel` is defined just like ZeroMQ. In-process channels can be defined as `ipc://<name>` or TCP ports can be defined as `tcp://<host>:<port>`.
* Each slab is a contiguous allocation of memory, so in-process communication only signals the bump pointer to the slot header for the message.
* Depending on the channel type, the slab can be defined as:
  - Single Producer, Single Consumer
  - Single Producer, Multi Consumer
  - Multi Producer, Single Consumer
  - Multi Producer, Multi Consumer

---

## 2. Slab Wire Format

| Offset | Bytes | Field               | Notes                      |
| -----: | ----: | ------------------- | -------------------------- |
|      0 |     4 | **len**             | Total bytes after this u32 |
|      4 |     4 | *hash* *(future)*   | xxhash32 (for tcp only)    |
|   4+32 |     8 | **VariantHdr** (1)  | type metadata for casting  |
|      … |     n | payload for hdr (1) | 8‑byte aligned             |
|      … |     8 | VariantHdr (2)      | repeated if multiple types |
|      … |     m | payload (2)         |                            |

### `VariantHdr` (8 bytes, 8‑byte aligned)

```cpp
struct VariantHdr {
    uint8_t  type;
    uint8_t  flags;
    uint16_t reserved;
    uint32_t byteLen;
};
```

---

### Memory‑View API

Since the types are defined by the `VariantHdr`, we can use a memory‑view API with templated `VariantView` classes to access the data in the slab without copying it. This should be done in a way that is as easy to use as possible, so projects that include this library only need minimal lines of code to view and manipulate the data.

---

## 4. Network Layer

The project currently uses Boost.Asio for networking.

