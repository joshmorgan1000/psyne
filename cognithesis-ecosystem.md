# Cognithesis Ecosystem Overview

## Vision
Democratize neural network design and usage, making AI accessible to everyone - from children to experts - through intuitive tools and powerful infrastructure.

## Core Components

### 1. Harmonics (Neural Network Engine)
- **Purpose**: Execute neural networks defined in the Harmonics DSL
- **Language**: C++20 header-only library
- **Key Features**:
  - Stream-based computation model
  - Substrate-agnostic (CPU/GPU/future quantum)
  - Runtime precision negotiation
  - Apple Silicon optimized (primary target)
  - Model import/export (ONNX, GGUF, TensorFlow, etc.)
- **Dependencies**: Eigen (linear algebra), Psyne (transport)
- **MVP Goal**: Run a 7B parameter LLM from HuggingFace

### 2. Psyne (High-Performance Messaging)
- **Purpose**: Zero-copy tensor transport between neural network layers
- **Language**: C++
- **Key Features**:
  - Lock-free channels (SPSC, MPSC, SPMC, MPMC)
  - Multiple transports (memory, IPC, TCP, unix sockets)
  - Arrow protocol support (for Spark integration)
  - Ring buffer based for maximum performance
- **Status**: Implemented and functional

### 3. ManifolDB (Distributed Storage)
- **Purpose**: Horizontally scalable distributed hash table for massive model storage
- **Key Features**:
  - Outperforms existing solutions (ScyllaDB, etc.)
  - Designed for distributed neural network graphs
  - Handles millions of concurrent users
- **Status**: Functional, undergoing code review and upgrades

### 4. Simplex-PDX (Vector Search)
- **Purpose**: High-performance similarity search for embeddings
- **Algorithm**: PDX-BOND with proprietary optimizations
- **Key Features**:
  - Handles 4000+ dimensional vectors efficiently
  - Powers the libraIan natural language interface
- **Status**: Implemented and functional

### 5. libraIan (Natural Language Interface)
- **Purpose**: Allow users to "talk" to their data through natural language
- **Key Features**:
  - Universal autoencoder for data representation
  - Maps natural language to latent space
  - Enables queries like "show me happy customers"
- **Status**: Conceptual

### 6. Cognithesis Application
- **Purpose**: Unified UI bringing all components together
- **UI Style**: Similar to Rete.js (node-based visual programming)
- **Key Features**:
  - Visual neural network design
  - Real-time training visualization
  - Performance metrics and tuning
  - Scales from single machine to distributed clusters
- **Status**: In development

## Integration Architecture

```
┌─────────────────────────────────────────────┐
│          Cognithesis UI (Rete.js-style)     │
├─────────────────────────────────────────────┤
│  libraIan         │  Training/Inference     │
│  (NLP Interface)  │  Management             │
├───────────────────┴─────────────────────────┤
│            Harmonics DSL Engine             │
├─────────────────────────────────────────────┤
│   Psyne      │   ManifolDB   │  Simplex-PDX │
│  (Transport) │   (Storage)   │   (Search)   │
└─────────────────────────────────────────────┘
```

## Development Philosophy

1. **No Python dependency** - Pure C++ for performance and deployment simplicity
2. **Start simple, scale later** - Browser JS → Native → Distributed
3. **Zero-copy everywhere** - Minimize data movement
4. **Substrate agnostic** - Same code runs on phone or supercomputer
5. **Democratize AI** - Make neural networks as easy as spreadsheets

## Proof of Concept: Chain Beasts

A game demonstrating how Cognithesis makes AI tangible:
- Neural networks as game characters
- On-chain training with proof-of-useful-work
- Demonstrates the full stack working together
- Gateway to get users interested in AI

*Note: Chain Beasts is a separate project that will utilize Harmonics once the core engine is complete.*

## Roadmap Progression

1. **Phase 1**: Get Chain Beasts working (JavaScript + basic Harmonics)
2. **Phase 2**: Native performance (full Harmonics + Psyne)
3. **Phase 3**: Visual tools (Cognithesis UI)
4. **Phase 4**: Scale out (ManifolDB + Spark integration)
5. **Phase 5**: Natural language (libraIan)

## Success Metrics

- **Technical**: Run Llama 2 7B on consumer hardware
- **Usability**: Child can design and train a neural network
- **Adoption**: 1M+ users training custom models
- **Impact**: AI becomes a tool everyone can use, not just experts