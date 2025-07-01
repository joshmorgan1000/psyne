/**
 * @file zero_copy_gpu_example.cpp
 * @brief Example of zero-copy GPU communication using Psyne GPU channels
 *
 * Demonstrates how to:
 * 1. Allocate messages directly in GPU-accessible memory
 * 2. Launch GPU kernels on message data without copies
 * 3. Share results back to CPU with zero copy
 */

#include "logger.hpp"
#include "psyne/channel/gpu_channel.hpp"
#include "psyne/psyne.hpp"
#include <chrono>
#include <vector>

using namespace psyne;

// Example message for GPU processing
struct TensorMessage {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t batch_size;
    float data[]; // Variable-sized tensor data

    size_t tensor_size() const {
        return width * height * channels * batch_size * sizeof(float);
    }
};

#ifdef PSYNE_CUDA_ENABLED
// Simple CUDA kernel for tensor processing
__global__ void process_tensor_kernel(float *data, uint32_t size,
                                      float scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Simple operation: multiply by scalar and add index
        data[idx] = data[idx] * scalar + idx * 0.001f;
    }
}
#endif

// Producer: Creates tensors and sends for GPU processing
void tensor_producer(std::shared_ptr<GPUChannel> channel, size_t num_tensors,
                     uint32_t width, uint32_t height) {
    log_info("Producer: Creating ", num_tensors, " tensors of size ", width,
             "x", height);

    for (size_t i = 0; i < num_tensors; ++i) {
        // Calculate tensor size
        size_t tensor_elements = width * height;
        size_t data_size = tensor_elements * sizeof(float);
        size_t total_size = sizeof(TensorMessage) + data_size;

        // Allocate message directly in GPU memory
        auto msg = channel->allocate(total_size);

        // Set tensor metadata
        msg->width = width;
        msg->height = height;
        msg->channels = 1;
        msg->batch_size = 1;

        // Initialize tensor data on CPU
        float *data = msg->data;
        for (size_t j = 0; j < tensor_elements; ++j) {
            data[j] = static_cast<float>(i * tensor_elements + j);
        }

        // The data is already in GPU-accessible memory!
        // No need to copy when we send
        msg.send();

        if (i % 100 == 0 && i > 0) {
            log_info("Producer: Sent ", i, " tensors");
        }
    }

    log_info("Producer: Finished sending ", num_tensors, " tensors");
}

// Consumer: Receives tensors and processes them on GPU
void tensor_consumer(std::shared_ptr<GPUChannel> channel, size_t num_tensors,
                     uint32_t width, uint32_t height) {
    log_info("Consumer: Ready to process tensors on GPU");

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_tensors; ++i) {
        // Receive tensor
        auto msg = channel->receive();

        // Get GPU pointer - no copy needed!
        float *gpu_data = static_cast<float *>(msg.gpu_ptr()) +
                          sizeof(TensorMessage) / sizeof(float);

        size_t tensor_elements = msg->width * msg->height;

#ifdef PSYNE_CUDA_ENABLED
        if (channel->get_backend() == GPUBackend::CUDA) {
            // Launch CUDA kernel directly on the message data
            int block_size = 256;
            int grid_size = (tensor_elements + block_size - 1) / block_size;

            process_tensor_kernel<<<grid_size, block_size>>>(
                gpu_data, tensor_elements, 2.0f);

            // Ensure GPU writes are visible to CPU
            msg.sync_to_host();
        }
#endif

#ifdef PSYNE_METAL_ENABLED
        if (channel->get_backend() == GPUBackend::METAL) {
            // Metal compute shader would go here
            log_info("Metal compute not implemented in this example");
        }
#endif

        // Verify results (reading from GPU memory)
        if (i == 0) {
            float *cpu_data = msg->data;
            log_info("First tensor after GPU processing:");
            log_info("  data[0] = ", cpu_data[0]);
            log_info("  data[1] = ", cpu_data[1]);
            log_info("  data[", tensor_elements - 1,
                     "] = ", cpu_data[tensor_elements - 1]);
        }

        // Release message
        channel->release(msg);

        if (i % 100 == 0 && i > 0) {
            log_info("Consumer: Processed ", i, " tensors");
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

    log_info("Consumer: Processed ", num_tensors, " tensors in ", duration,
             " ms");

    double throughput_gb = (num_tensors * width * height * sizeof(float)) /
                           (1024.0 * 1024.0 * 1024.0) / (duration / 1000.0);
    log_info("Throughput: ", throughput_gb, " GB/s");
}

// Multi-GPU example
void multi_gpu_example() {
    log_info("=== Multi-GPU Zero-Copy Example ===");

#ifdef PSYNE_CUDA_ENABLED
    int device_count;
    cudaGetDeviceCount(&device_count);

    if (device_count < 2) {
        log_info("Need at least 2 GPUs for multi-GPU example");
        return;
    }

    // Create channels on different GPUs
    GPUChannelConfig config1;
    config1.name = "gpu0_channel";
    config1.size_mb = 64;
    config1.device_id = 0;
    config1.enable_peer_access = true;

    GPUChannelConfig config2;
    config2.name = "gpu1_channel";
    config2.size_mb = 64;
    config2.device_id = 1;
    config2.enable_peer_access = true;

    auto channel1 = std::make_shared<GPUChannel>(config1);
    auto channel2 = std::make_shared<GPUChannel>(config2);

    // Check if zero-copy is possible between GPUs
    if (channel1->can_zero_copy_with(*channel2)) {
        log_info("GPU 0 and GPU 1 can do peer-to-peer transfers!");

        // Enable peer access
        channel1->enable_peer_access(1);
        channel2->enable_peer_access(0);

        // Now GPU 0 can directly access GPU 1's memory and vice versa
        log_info("Peer access enabled for zero-copy GPU-to-GPU transfers");
    } else {
        log_info("Direct GPU-to-GPU transfer not supported");
    }
#else
    log_info("CUDA not available for multi-GPU example");
#endif
}

int main(int argc, char *argv[]) {
    try {
        log_info("Psyne GPU Zero-Copy Example");
        log_info("===========================");

        // Configuration
        size_t num_tensors = 1000;
        uint32_t tensor_width = 1024;
        uint32_t tensor_height = 1024;

        if (argc > 1)
            num_tensors = std::stoul(argv[1]);
        if (argc > 2)
            tensor_width = std::stoul(argv[2]);
        if (argc > 3)
            tensor_height = std::stoul(argv[3]);

        // Create GPU channel
        GPUChannelConfig config;
        config.name = "tensor_processing_channel";
        config.size_mb = 256; // 256MB buffer
        config.mode = ChannelMode::SPSC;
        config.blocking = true;
        config.backend = GPUBackend::AUTO; // Auto-select best GPU backend

#ifdef PSYNE_CUDA_ENABLED
        // Prefer unified memory for simplicity
        config.memory_flags =
            static_cast<uint32_t>(GPUMemoryFlags::UNIFIED_MEMORY);
#else
        // Use host-visible coherent memory
        config.memory_flags =
            static_cast<uint32_t>(GPUMemoryFlags::HOST_VISIBLE) |
            static_cast<uint32_t>(GPUMemoryFlags::HOST_COHERENT);
#endif

        auto channel = std::make_shared<GPUChannel>(config);

        std::string backend_name;
        switch (channel->get_backend()) {
        case GPUBackend::CUDA:
            backend_name = "CUDA";
            break;
        case GPUBackend::METAL:
            backend_name = "Metal";
            break;
        case GPUBackend::VULKAN:
            backend_name = "Vulkan";
            break;
        default:
            backend_name = "Unknown";
        }
        log_info("Created GPU channel with backend: ", backend_name);

        // Run producer and consumer
        std::thread producer_thread(tensor_producer, channel, num_tensors,
                                    tensor_width, tensor_height);
        std::thread consumer_thread(tensor_consumer, channel, num_tensors,
                                    tensor_width, tensor_height);

        producer_thread.join();
        consumer_thread.join();

        // Print statistics
        auto stats = channel->get_stats();
        log_info("Channel Statistics:");
        log_info("  Messages sent: ", stats.messages_sent);
        log_info("  Messages received: ", stats.messages_received);
        log_info("  Total data: ",
                 (stats.bytes_sent + stats.bytes_received) /
                     (1024.0 * 1024.0 * 1024.0),
                 " GB");
        log_info("  Average latency: ", stats.avg_latency_ns / 1000.0, " µs");

        // Run multi-GPU example if available
        multi_gpu_example();

    } catch (const std::exception &e) {
        log_error("Error: ", e.what());
        return 1;
    }

    return 0;
}