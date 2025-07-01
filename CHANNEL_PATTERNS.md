# Psyne Channel Patterns Design

## Overview

This document describes the design and implementation of various channel patterns in Psyne, optimized for different neural network training scenarios.

## Channel Patterns

### 1. SPSC (Single Producer Single Consumer) - ✅ Implemented
**Use Case**: Layer-to-layer forward pass, sequential processing
**Performance**: Highest (lock-free, no contention)
**Implementation**: Wait-free ring buffer with cached positions

### 2. MPSC (Multiple Producer Single Consumer)
**Use Case**: Gradient aggregation, multiple workers → parameter server
**Performance**: High (lock-free producers, single consumer)
**Algorithm**: Per-producer slots with atomic claim mechanism

#### Design
```cpp
// MPSC Layout
[Producer Slots Array]
  - N fixed-size slots (N = max producers)
  - Each slot has atomic state: EMPTY, WRITING, READY
  
[Consumer State]
  - Next slot to read
  - Cached producer states
  
[Slot Structure]
  - Atomic state
  - Message data
  - Producer ID
```

#### Use Cases in ML
- **Gradient Aggregation**: Multiple backward passes → single optimizer
- **Data Loading**: Multiple data loaders → single training thread
- **Metrics Collection**: Multiple workers → single logger

### 3. SPMC (Single Producer Multiple Consumer)
**Use Case**: Data parallel broadcast, batch splitting
**Performance**: Medium (atomic read position updates)
**Algorithm**: Broadcast ring buffer with reference counting

#### Design
```cpp
// SPMC Layout
[Producer State]
  - Write position
  - Reference count per message
  
[Consumer Registry]
  - Consumer count
  - Per-consumer read positions
  
[Message Lifecycle]
  - Producer writes and sets ref count = num_consumers
  - Each consumer decrements ref count on read
  - Space freed when ref count = 0
```

#### Use Cases in ML
- **Data Parallelism**: Broadcast batches to multiple GPUs
- **Model Parallelism**: Fan-out activations to parallel layers
- **Ensemble Training**: Same data to multiple models

### 4. MPMC (Multiple Producer Multiple Consumer)
**Use Case**: Work queue, general purpose
**Performance**: Lower (requires synchronization)
**Algorithm**: Lock-free queue with CAS operations

#### Design
```cpp
// MPMC Layout
[Ring Buffer]
  - Array of atomic slots
  - Each slot: {sequence, data}
  
[Atomic Positions]
  - Head (for consumers)
  - Tail (for producers)
  
[Algorithm]
  - Producers CAS on tail
  - Consumers CAS on head
  - Sequence numbers prevent ABA problem
```

#### Use Cases in ML
- **Dynamic Batching**: Multiple data sources → multiple trainers
- **Federated Learning**: Peer-to-peer gradient exchange
- **Pipeline Parallelism**: Flexible stage connections

## Implementation Strategy

### 1. MPSC Implementation
```cpp
class MPSCChannel : public ChannelBase {
    struct Slot {
        std::atomic<SlotState> state{EMPTY};
        alignas(64) char data[MAX_MESSAGE_SIZE];
        size_t size;
        uint32_t producer_id;
    };
    
    // Fixed array of slots
    std::vector<Slot> slots_;
    
    // Producer claims a slot atomically
    Slot* claim_slot(uint32_t producer_id);
    
    // Consumer processes slots in order
    Message* try_consume();
};
```

### 2. SPMC Implementation
```cpp
class SPMCChannel : public ChannelBase {
    struct Message {
        std::atomic<uint32_t> ref_count;
        MessageHeader header;
        // ... data
    };
    
    // Consumers register for broadcasts
    void register_consumer(uint32_t consumer_id);
    
    // Producer broadcasts to all
    void broadcast(Message* msg);
    
    // Consumer receives their copy
    Message* receive(uint32_t consumer_id);
};
```

### 3. MPMC Implementation
```cpp
class MPMCChannel : public ChannelBase {
    struct Cell {
        std::atomic<size_t> sequence;
        Message* data;
    };
    
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    std::vector<Cell> buffer_;
    
    // Lock-free enqueue/dequeue
    bool enqueue(Message* msg);
    Message* dequeue();
};
```

## Performance Considerations

### Cache Line Optimization
- Each producer/consumer gets dedicated cache line
- Padding to prevent false sharing
- Aligned allocations for SIMD operations

### Memory Ordering
- **MPSC**: Producers use release, consumer uses acquire
- **SPMC**: Producer uses release, consumers use acquire-release
- **MPMC**: Full sequential consistency for simplicity

### Backoff Strategies
- Exponential backoff for contended operations
- PAUSE instruction in spin loops
- Adaptive spinning based on contention level

## Example Usage

### MPSC - Gradient Aggregation
```cpp
auto channel = Channel<GradientMessage>::create({
    .mode = ChannelMode::MPSC,
    .size_mb = 64,
    .max_producers = num_workers
});

// Workers
for (int worker = 0; worker < num_workers; ++worker) {
    threads.emplace_back([&, worker] {
        auto grad = channel->allocate(worker);
        compute_gradients(grad->data());
        grad->send();
    });
}

// Parameter server
while (auto grad = channel->receive()) {
    aggregate_gradients(grad->data());
}
```

### SPMC - Data Parallel Training
```cpp
auto channel = Channel<BatchMessage>::create({
    .mode = ChannelMode::SPMC,
    .size_mb = 128,
    .num_consumers = num_gpus
});

// Data loader
while (auto batch = load_batch()) {
    auto msg = channel->allocate();
    *msg = batch;
    msg->broadcast();
}

// GPU workers
for (int gpu = 0; gpu < num_gpus; ++gpu) {
    threads.emplace_back([&, gpu] {
        channel->register_consumer(gpu);
        while (auto batch = channel->receive(gpu)) {
            train_on_gpu(gpu, batch->data());
        }
    });
}
```

### MPMC - Work Queue
```cpp
auto channel = Channel<TaskMessage>::create({
    .mode = ChannelMode::MPMC,
    .size_mb = 32
});

// Multiple producers
for (auto& producer : producers) {
    auto task = channel->allocate();
    task->fill_work();
    task->send();
}

// Multiple consumers  
for (auto& worker : workers) {
    while (auto task = channel->receive()) {
        process_task(task->data());
    }
}
```

## Testing Strategy

1. **Correctness Tests**
   - Message ordering guarantees
   - No message loss under contention
   - Proper cleanup and memory management

2. **Performance Tests**
   - Throughput under various producer/consumer ratios
   - Latency distribution
   - Scalability with core count

3. **Stress Tests**
   - High contention scenarios
   - Rapid producer/consumer creation/destruction
   - Memory pressure conditions