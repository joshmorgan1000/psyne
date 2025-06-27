# Psyne RPC – Philosophy & Roadmap

> **Scope** — This document covers only the *RPC layer* of Psyne: slab‑backed message‑passing designed for high‑throughput vector workloads.

---

## 1 · Philosophy

| Pillar                                  | Rationale                                                                                                                                                       |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Memory‑first, network‑second**        | The fastest message is one that never leaves device memory.  All transport abstractions ultimately map to GPU‑/host‑visible Vulkan buffers ("slabs").           |
| **Contiguous rings > fragmented heaps** | Single–producer/consumer (SPSC) ring buffers eliminate locks and minimise cache‑misses. Higher‑level patterns (MPMC, pub/sub) are composed atop multiple rings. |
| **Variant views instead of copies**     | `VariantView<T, Rank>` lets users reinterpret bytes in‑place; no per‑message allocations.                                                                       |
| **Asynchronous by default**             | `co_await`–based APIs expose *composable* coroutines, hiding reactor plumbing yet enabling back‑pressure.                                                       |
| **Client‑defined back‑pressure**        | The sender allocates its response slab; the server merely writes into it. Throughput & latency become explicit API choices.                                     |
| **Zero secrets on the wire**            | All request IDs are salted/hashes; encryption can be re‑enabled later without changing the wire layout.                                                         |

---

## 2 · Current Architecture (MVP‑0)

```text
         ┌──────────┐    SPSC   ┌──────────────┐
Client ─►│ Req Slab │──────────►│Broker Thread │
         └──────────┘           │  (echo/route)│
                                └────▲────┬────┘
                                     │    │
                                     │ SPSC
                                     │    │
                               ┌─────┴─────▼────┐
                               │Shard Worker(s) │
                               └────────────────┘
```

* **Rings:** Two SPSC slabs per logical connection (request/response).
* **Broker:** Single‑thread, single input ring → scatter requests to shard rings, no response aggregation yet.
* **Worker:** One shard == one hash range; picks from its input ring, writes to caller’s response ring.
* **API Surface:** `Socket::bind | connect | send_bytes | receive_bytes` + helpers `send_vector`, etc.

---

## 3 · Roadmap

### 0‑3 months — *MVP hardening*

| Item                        | Notes                                                                 |
| --------------------------- | --------------------------------------------------------------------- |
| **Finish SPSC API**         | Ensure `bind/connect` semantics are intuitive; polish VariantBuilder. |
| **Broker batching**         | Copy‑less scatter: chunked memcpy into shard rings.                   |
| **Back‑pressure telemetry** | Slab fill‑level metrics exported via Prom‑style counters.             |
| **Unit & soak tests**       | Stress up to ≥ 10 M msg/s, verify no leaks / over‑runs on ASan.       |

### 3‑9 months — *Scale‑out & Ergonomics*

| Item                       | Notes                                                                          |
| -------------------------- | ------------------------------------------------------------------------------ |
| **MPMC ring option**       | Replace duplicated rings with single multi‑producer variant where beneficial.  |
| **Network transports**     | IPC → TCP → RDMA; abstract via `Transport` trait.                              |
| **Encrypted slabs**        | AES‑GCM inline; optional per‑channel keys.                                     |
| **Schema‑driven code‑gen** | Declarative message schemas → templated getters/setters, zero reflection cost. |
| **C / Python bindings**    | C‑FFI first, then PyBind11 thin‑wrapper for rapid prototyping.                 |

### > 9 months — *Intelligence & Autonomy*

| Horizon                           | Idea                                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------------------ |
| **Self‑tuning slab sizes**        | Broker records watermarks, resizes (or spawns new rings) at runtime.                       |
| **Typed pub/sub**                 | Topic hash → ring mapping; vector search results fanned‑out to subscribers.                |
| **Co‑scheduled GPU kernels**      | Workers enqueue compute shaders directly on slabs for in‑GPU vector ops.                   |
| **Cluster coordinator (k8s‑CRD)** | Declarative shards/brokers; slabs become shared memory segments across pods via GPU IOMMU. |

---

## 4 · Guiding Questions

1. *What is the highest sustained msgs/s on commodity hardware?*
2. *When does SPSC → MPMC crossover pay off?* (measure!)
3. *How do we expose just‑enough knobs without overwhelming users?*
4. *Can we keep the zero‑copy promise even across the network?*

---

## 5 · Contributing

* **Style:** clang‑format, C++20, no exceptions in hot‑path.
* **Testing:** GoogleTest + ASan/TSan, perf tests under `tests/`.
* **Docs:** Any design change → update this roadmap.

> “Fast systems are grown, not built.” — Psyne mantra

## DSL Roadmap (Harmonics DSL)

### Why a DSL?

* **Express AI pipelines declaratively** – chains of vector operations, reductions, and model invocations.
* **Self‑balancing execution** – scheduler can reorder, fuse, or split ops across shards & GPUs while respecting data‑dependencies.
* **Portability** – same description targets CPU fallback, Vulkan compute, or CUDA.

### Minimal MVP

1. **Typed byte‑code**: `OP_LOAD`, `OP_DOT`, `OP_SUM`, `OP_ACT`, … 8‑byte header + operands, packed into slabs so programs stream like messages.
2. **Variant‑aware**: ops accept/produce Variant handles, zero‑copy across rings.
3. **Interpreter kernel**: single‑pass interpreter in GLSL/SPIR‑V that walks byte‑code stored in a Vulkan SSBO.

### Milestones

| Phase | Deliverable                    | Notes                                   |
| ----- | ------------------------------ | --------------------------------------- |
| 0     | Text→bytecode assembler in C++ | Unit‑tests only                         |
| 1     | CPU interpreter (for CI)       | Uses same Variant views                 |
| 2     | Vulkan compute interpreter     | One warp ≈ one op pipeline              |
| 3     | Static optimiser pass          | constant‑fold, op‑fusion                |
| 4     | Self‑balancing runtime         | feedback loops adjust shard/batch sizes |

### Long‑term stretch

* **Auto‑diff & scheduling for training loops**.
* **JIT kernels** for rare heavy ops.
* **Graph‑level pub/sub** – DSL programs themselves become messages.
