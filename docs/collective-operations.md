# Collective Operations in Psyne

## Overview

Psyne's collective operations provide high-performance communication patterns essential for distributed computing and AI/ML workloads. These operations enable efficient data exchange among multiple processes, forming the backbone of distributed training, parallel algorithms, and HPC applications.

## Key Features

- **Ring-based algorithms** for optimal bandwidth utilization
- **Zero-copy message passing** leveraging psyne's core architecture
- **Multiple reduction operations** (sum, product, min, max, bitwise)
- **Async execution** with futures for non-blocking operations
- **Flexible topology support** (ring, tree, mesh)

## Supported Operations

### 1. Broadcast
Send data from one process (root) to all other processes.

```cpp
// Root process (rank 0) broadcasts data to all others
std::vector<float> data = {1.0f, 2.0f, 3.0f};
Broadcast<float> bcast(group);
bcast.execute(std::span<float>(data), root_rank);
```

**Use cases:**
- Distributing model parameters
- Sharing configuration data
- Broadcasting control signals

### 2. All-Reduce
Perform a reduction operation across all processes and distribute the result to everyone.

```cpp
// Sum gradients across all workers
std::vector<float> gradients(1000);
AllReduce<float> allreduce(group);
allreduce.execute(std::span<float>(gradients), ReduceOp::Sum);

// Average by dividing by world size
for (auto& g : gradients) g /= group->size();
```

**Use cases:**
- Gradient aggregation in distributed training
- Global statistics computation
- Consensus algorithms

### 3. Scatter
Distribute different chunks of data from root to all processes.

```cpp
// Root scatters chunks to all processes
std::vector<int> all_data = {1,2,3,4,5,6,7,8,9,10,11,12};
std::vector<int> my_chunk(4); // Each process gets 4 elements

Scatter<int> scatter(group);
scatter.execute(std::span<const int>(all_data), 
                std::span<int>(my_chunk), root_rank);
```

**Use cases:**
- Data parallelism
- Distributing dataset chunks
- Work distribution

### 4. Gather
Collect data from all processes to root.

```cpp
// Each process sends its data to root
std::vector<float> local_result = compute_local_result();
std::vector<float> all_results; // Only used by root

if (rank == root) {
    all_results.resize(world_size * local_result.size());
}

Gather<float> gather(group);
gather.execute(std::span<const float>(local_result),
               std::span<float>(all_results), root_rank);
```

**Use cases:**
- Result collection
- Centralized logging
- Model checkpointing

### 5. All-Gather
Gather data from all processes and distribute to everyone.

```cpp
// Each process contributes local data
std::vector<double> local_stats = {mean, variance};
std::vector<double> all_stats(world_size * 2);

AllGather<double> allgather(group);
allgather.execute(std::span<const double>(local_stats),
                  std::span<double>(all_stats));
```

**Use cases:**
- Sharing model states
- Distributed consensus
- Global view synchronization

## Ring Algorithms

Psyne implements efficient ring-based algorithms for collective operations:

### Ring All-Reduce
1. **Reduce-scatter phase**: Each process reduces a specific chunk
2. **All-gather phase**: Share reduced chunks with all processes

This achieves optimal bandwidth utilization: `(2 * (N-1) * data_size) / N` bytes transferred per node.

### Benefits:
- **Bandwidth optimal**: Uses full bandwidth of all links
- **Load balanced**: Equal work distribution
- **Scalable**: O(N) communication steps

## Performance Optimization

### 1. Transport Selection
```cpp
// Use high-performance transports for large clusters
auto group = create_collective_group(rank, {
    "rdma://node1:5000",  // InfiniBand for HPC
    "tcp://node2:5000",   // TCP fallback
    "ipc://shared_mem"    // Shared memory for local
});
```

### 2. Overlapping Computation
```cpp
// Start async collective
auto future = allreduce.execute_async(gradients, ReduceOp::Sum);

// Do other work while collective runs
compute_forward_pass();

// Wait for collective to complete
future.wait();
```

### 3. Chunked Operations
For very large data, process in chunks to improve cache efficiency:

```cpp
const size_t chunk_size = 1024 * 1024; // 1MB chunks
for (size_t offset = 0; offset < data.size(); offset += chunk_size) {
    size_t count = std::min(chunk_size, data.size() - offset);
    auto chunk = std::span<float>(data.data() + offset, count);
    allreduce.execute(chunk, ReduceOp::Sum);
}
```

## ML/AI Integration

### Distributed Training Example
```cpp
// Training loop with gradient aggregation
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    // Forward pass
    auto loss = model.forward(batch);
    
    // Backward pass - compute local gradients
    auto gradients = model.backward(loss);
    
    // All-reduce gradients across all workers
    AllReduce<float> allreduce(group);
    allreduce.execute(gradients, ReduceOp::Sum);
    
    // Average gradients
    for (auto& g : gradients) g /= world_size;
    
    // Update model parameters
    optimizer.step(gradients);
    
    // Synchronize before next iteration
    group->barrier();
}
```

### Data Parallel Training
```cpp
// Scatter dataset to workers
std::vector<DataBatch> all_batches = load_dataset();
DataBatch my_batch;

Scatter<DataBatch> scatter(group);
scatter.execute(all_batches, my_batch, 0);

// Train on local batch
auto local_loss = train_on_batch(my_batch);

// Gather losses for monitoring
std::vector<float> all_losses;
Gather<float> gather(group);
gather.execute(local_loss, all_losses, 0);
```

## Advanced Patterns

### 1. Reduce-Scatter
Combine reduce and scatter in one operation:
```cpp
// Each process gets a reduced chunk
template<typename T>
void reduce_scatter(std::span<T> data, CollectiveGroup& group, ReduceOp op) {
    const size_t chunk_size = data.size() / group.size();
    const size_t my_chunk = group.rank() * chunk_size;
    
    // First do all-reduce
    AllReduce<T> allreduce(group);
    allreduce.execute(data, op);
    
    // Then each process keeps only its chunk
    std::vector<T> my_data(data.begin() + my_chunk, 
                          data.begin() + my_chunk + chunk_size);
    data = my_data;
}
```

### 2. Pipeline Parallelism
```cpp
// Pipeline stages on different ranks
if (rank == 0) {
    auto data = load_input();
    channel_to_next->send(data);
} else if (rank < world_size - 1) {
    auto data = channel_from_prev->receive();
    auto processed = process_stage(data);
    channel_to_next->send(processed);
} else {
    auto data = channel_from_prev->receive();
    auto result = final_stage(data);
    save_output(result);
}
```

### 3. Hierarchical Collectives
For large clusters, use hierarchical operations:
```cpp
// Two-level hierarchy: local node + across nodes
auto local_group = create_collective_group(local_rank, local_peers);
auto global_group = create_collective_group(node_rank, node_peers);

// Local reduction first
allreduce.execute(data, ReduceOp::Sum, local_group);

// Then global reduction (only one rank per node)
if (local_rank == 0) {
    allreduce.execute(data, ReduceOp::Sum, global_group);
}

// Broadcast back to local ranks
broadcast.execute(data, 0, local_group);
```

## Error Handling

```cpp
try {
    allreduce.execute(data, ReduceOp::Sum);
} catch (const std::runtime_error& e) {
    // Handle communication errors
    std::cerr << "Collective failed: " << e.what() << std::endl;
    
    // Attempt recovery
    group->barrier(); // Resynchronize
    retry_with_fallback_transport();
}
```

## Best Practices

1. **Choose the right algorithm**: Ring for bandwidth-bound, tree for latency-bound
2. **Minimize synchronization**: Use async operations when possible
3. **Optimize data layout**: Ensure contiguous memory for better performance
4. **Profile and measure**: Use psyne's metrics to identify bottlenecks
5. **Handle failures gracefully**: Implement retry logic and fallbacks

## Future Enhancements

- **GPU-aware collectives**: Direct GPU memory operations
- **Compression**: Gradient compression for bandwidth reduction
- **Fault tolerance**: Automatic recovery from node failures
- **Dynamic groups**: Add/remove processes during execution