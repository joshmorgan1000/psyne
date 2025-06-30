/**
 * @file vulkan_gpu_demo.cpp
 * @brief Demonstrates Vulkan GPU support
 */

#include "../src/gpu/gpu_buffer.hpp"
#include "../src/gpu/gpu_context.cpp"
#include <chrono>
#include <iostream>
#include <psyne/psyne.hpp>
#include <vector>

using namespace psyne;
using namespace psyne::gpu;
using namespace std::chrono;

int main() {
    std::cout << "Vulkan GPU Demo\n";
    std::cout << "===============\n\n";

    // Detect available GPU backends
    auto backends = detect_gpu_backends();

    std::cout << "Available GPU backends:\n";
    for (auto backend : backends) {
        std::cout << "  - " << gpu_backend_name(backend) << "\n";
    }

    // Check if Vulkan is available
    bool has_vulkan = false;
    for (auto backend : backends) {
        if (backend == GPUBackend::Vulkan) {
            has_vulkan = true;
            break;
        }
    }

    if (!has_vulkan) {
        std::cout << "\nVulkan not available on this system.\n";
        std::cout << "Make sure you have:\n";
        std::cout << "  - Vulkan drivers installed\n";
        std::cout << "  - A Vulkan-capable GPU\n";
        return 1;
    }

    // Create Vulkan context
    std::cout << "\nCreating Vulkan context...\n";
    auto context = create_gpu_context(GPUBackend::Vulkan);
    if (!context) {
        std::cerr << "Failed to create Vulkan context\n";
        return 1;
    }

    // Display GPU info
    std::cout << "GPU: " << context->device_name() << "\n";
    std::cout << "Total memory: " << context->total_memory() / (1024 * 1024)
              << " MB\n";
    std::cout << "Unified memory: "
              << (context->is_unified_memory() ? "Yes" : "No") << "\n\n";

    // Create buffer factory
    auto factory = context->create_buffer_factory();
    if (!factory) {
        std::cerr << "Failed to create buffer factory\n";
        return 1;
    }

    // Test 1: Basic buffer allocation
    std::cout << "Test 1: Basic Buffer Allocation\n";
    const size_t buffer_size = 16 * 1024 * 1024; // 16MB

    auto buffer = factory->create_buffer(buffer_size, BufferUsage::Dynamic,
                                         MemoryAccess::Shared);

    if (!buffer) {
        std::cerr << "Failed to create buffer\n";
        return 1;
    }

    std::cout << "  Created " << buffer->size() / (1024 * 1024)
              << " MB buffer\n";
    std::cout << "  Usage: "
              << (buffer->usage() == BufferUsage::Dynamic ? "Dynamic"
                                                          : "Static")
              << "\n";
    std::cout << "  Access: "
              << (buffer->access() == MemoryAccess::Shared ? "Shared"
                                                           : "Device-only")
              << "\n\n";

    // Test 2: Data upload/download
    std::cout << "Test 2: Data Upload/Download\n";

    std::vector<float> test_data(1024 * 1024); // 1M floats
    for (size_t i = 0; i < test_data.size(); ++i) {
        test_data[i] = static_cast<float>(i);
    }

    // Upload data
    auto upload_start = high_resolution_clock::now();
    buffer->upload(test_data.data(), test_data.size() * sizeof(float));
    auto upload_end = high_resolution_clock::now();

    auto upload_time =
        duration_cast<microseconds>(upload_end - upload_start).count();
    double upload_bandwidth =
        (test_data.size() * sizeof(float) / 1024.0 / 1024.0) /
        (upload_time / 1e6);

    std::cout << "  Upload time: " << upload_time << " μs\n";
    std::cout << "  Upload bandwidth: " << upload_bandwidth << " MB/s\n";

    // Download data
    std::vector<float> downloaded(test_data.size());

    auto download_start = high_resolution_clock::now();
    buffer->download(downloaded.data(), downloaded.size() * sizeof(float));
    auto download_end = high_resolution_clock::now();

    auto download_time =
        duration_cast<microseconds>(download_end - download_start).count();
    double download_bandwidth =
        (downloaded.size() * sizeof(float) / 1024.0 / 1024.0) /
        (download_time / 1e6);

    std::cout << "  Download time: " << download_time << " μs\n";
    std::cout << "  Download bandwidth: " << download_bandwidth << " MB/s\n";

    // Verify data
    bool data_match = true;
    for (size_t i = 0; i < test_data.size(); ++i) {
        if (test_data[i] != downloaded[i]) {
            data_match = false;
            break;
        }
    }
    std::cout << "  Data verification: " << (data_match ? "PASSED" : "FAILED")
              << "\n\n";

    // Test 3: Memory mapping
    std::cout << "Test 3: Memory Mapping\n";

    void *mapped = buffer->map();
    if (mapped) {
        std::cout << "  Buffer mapped successfully\n";

        // Write directly to mapped memory
        float *mapped_floats = static_cast<float *>(mapped);
        for (size_t i = 0; i < 1000; ++i) {
            mapped_floats[i] = static_cast<float>(i * 2);
        }

        // Flush changes
        buffer->flush();
        buffer->unmap();

        std::cout << "  Direct memory write completed\n";
    } else {
        std::cout << "  Failed to map buffer\n";
    }

    // Test 4: Multiple buffers
    std::cout << "\nTest 4: Multiple Buffer Allocation\n";

    std::vector<std::unique_ptr<GPUBuffer>> buffers;
    const size_t num_buffers = 10;
    const size_t small_buffer_size = 1024 * 1024; // 1MB each

    auto multi_start = high_resolution_clock::now();

    for (size_t i = 0; i < num_buffers; ++i) {
        auto buf = factory->create_buffer(
            small_buffer_size, BufferUsage::Static, MemoryAccess::DeviceOnly);

        if (buf) {
            buffers.push_back(std::move(buf));
        }
    }

    auto multi_end = high_resolution_clock::now();
    auto multi_time =
        duration_cast<microseconds>(multi_end - multi_start).count();

    std::cout << "  Created " << buffers.size() << " buffers\n";
    std::cout << "  Total allocation time: " << multi_time << " μs\n";
    std::cout << "  Average per buffer: " << multi_time / buffers.size()
              << " μs\n";

    // Test 5: Performance with different access patterns
    std::cout << "\nTest 5: Access Pattern Performance\n";

    // Device-only buffer
    auto device_buffer = factory->create_buffer(
        buffer_size, BufferUsage::Static, MemoryAccess::DeviceOnly);

    // Shared buffer
    auto shared_buffer = factory->create_buffer(
        buffer_size, BufferUsage::Dynamic, MemoryAccess::Shared);

    if (device_buffer && shared_buffer) {
        // Test upload performance
        std::vector<float> perf_data(buffer_size / sizeof(float));

        // Device-only upload
        auto dev_start = high_resolution_clock::now();
        device_buffer->upload(perf_data.data(),
                              perf_data.size() * sizeof(float));
        context->synchronize();
        auto dev_end = high_resolution_clock::now();

        // Shared upload
        auto shared_start = high_resolution_clock::now();
        shared_buffer->upload(perf_data.data(),
                              perf_data.size() * sizeof(float));
        context->synchronize();
        auto shared_end = high_resolution_clock::now();

        auto dev_time =
            duration_cast<microseconds>(dev_end - dev_start).count();
        auto shared_time =
            duration_cast<microseconds>(shared_end - shared_start).count();

        std::cout << "  Device-only upload: " << dev_time << " μs\n";
        std::cout << "  Shared upload: " << shared_time << " μs\n";
        std::cout << "  Performance ratio: "
                  << static_cast<double>(shared_time) / dev_time << "x\n";
    }

    std::cout << "\nVulkan GPU support test completed successfully!\n";

    return 0;
}