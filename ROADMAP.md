# Psyne Roadmap

This document tracks short‑ and long‑term tasks.  Items are grouped roughly by
their expected timeline.  Details originate from the
[Philosophy](PHILOSOPHY.md) and [Messaging](MESSAGING.md) notes.

## 0‑3 months — MVP hardening

- [ ] Finish SPSC API and polish `VariantBuilder`.
- [ ] Broker batching for copy‑less scatter into shard rings.
- [ ] Back‑pressure telemetry via Prom‑style counters.
- [ ] Unit & soak tests to push ≥ 10 M msg/s under ASan.

## 3‑9 months — Scale‑out & Ergonomics

- [ ] MPMC ring option where multi‑producer queues pay off.
- [ ] Support multiple network transports (IPC → TCP → RDMA).
- [ ] Encrypted slabs with AES‑GCM and per‑channel keys.
- [ ] Schema‑driven code‑gen for zero‑cost getters/setters.
- [ ] C / Python bindings for rapid prototyping.

## > 9 months — Intelligence & Autonomy

- [ ] Self‑tuning slab sizes that adapt at runtime.
- [ ] Typed pub/sub with topic hashes and fan‑out.
- [ ] Co‑scheduled GPU kernels executing directly on slabs.
- [ ] Cluster coordinator (k8s‑CRD) to manage shards/brokers.

## In Progress

- [ ] Add `OP_PUT` and `OP_GET` operations to push tensors into the GPU ring.
- [ ] TLS support for the network layer.
- [ ] Multi‑backend GPU support (e.g., Metal on macOS).
