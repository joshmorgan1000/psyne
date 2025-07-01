# Psyne Documentation

Welcome to the Psyne documentation! Psyne is a high-performance, zero-copy messaging library designed for AI/ML workloads with a focus on neural network training and inference.

## Table of Contents

### Getting Started
- [Quick Start Guide](quick-start.md) - Get up and running in minutes
- [Basic Examples](../examples/) - Simple usage examples

### Core Concepts
- [Architecture Overview](architecture.md) - Core design principles and concepts
- [Protocols](protocols.md) - Data transformation layers

### Components
- [Examples](../examples/) - Comprehensive examples
- [Benchmarks](../benchmarks/) - Performance measurements

### Research & Attribution
- [TDT Compression](tdt_attribution.md) - Tensor compression research attribution

---

## Quick Navigation

| Component | Description | Example |
|-----------|-------------|---------|
| **Substrates** | Physical transport | `TCPSubstrate`, `InProcessSubstrate` |
| **Protocols** | Data transformation | `TDTCompressionProtocol`, `EncryptionProtocol` |
| **Patterns** | Coordination | `SPSC`, `MPSC`, `SPMC`, `MPMC` |
| **Messages** | Typed access | `TensorMessage`, `GradientMessage` |

## Key Features

✅ **Zero-Copy Performance** - No memory copies in the critical path  
✅ **Concept-Based Design** - Modern C++20 with no inheritance  
✅ **Plugin Architecture** - Easy to extend with custom components  
✅ **Multi-Transport** - In-process, IPC, TCP, UDP, GPU-direct  
✅ **ML Optimized** - Designed specifically for neural network workloads  
✅ **Cross-Platform** - Works on Linux, macOS, and Windows