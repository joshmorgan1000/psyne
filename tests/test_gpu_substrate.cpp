/**
 * @file test_gpu_substrate.cpp
 * @brief Basic test for GPU substrate functionality
 */

#include <psyne/channel/substrate/gpu.hpp>
#include <iostream>
#include <cstring>

int main() {
    std::cout << "=== GPU Substrate Test ===\n\n";
    
    // Check if any GPU backend is available
    auto backends = psyne::substrate::GPU::get_available_backends();
    if (backends.empty()) {
        std::cout << "No GPU backends available at compile time.\n";
        std::cout << "Test skipped.\n";
        return 0;  // Not an error - just no GPU support compiled in
    }
    
    std::cout << "Available GPU backends: ";
    for (const auto& backend : backends) {
        std::cout << backend << " ";
    }
    std::cout << "\n\n";
    
    try {
        // Test GPU substrate creation and memory allocation
        auto gpu = std::make_unique<psyne::substrate::GPU>();
        std::cout << "Created GPU substrate with backend: " 
                  << gpu->get_backend_name() << "\n";
        
        // Test memory allocation
        const size_t test_size = 1024 * 1024;  // 1MB
        void* mem = gpu->allocate_memory_slab(test_size);
        
        if (!mem) {
            std::cerr << "Failed to allocate GPU memory\n";
            return 1;
        }
        
        std::cout << "Successfully allocated " << test_size << " bytes of GPU memory\n";
        
        // Test memory access - write pattern
        std::memset(mem, 0x42, test_size);
        
        // Verify pattern
        unsigned char* bytes = static_cast<unsigned char*>(mem);
        bool success = true;
        for (size_t i = 0; i < test_size; ++i) {
            if (bytes[i] != 0x42) {
                success = false;
                break;
            }
        }
        
        if (success) {
            std::cout << "Memory read/write test passed\n";
        } else {
            std::cerr << "Memory read/write test failed\n";
            return 1;
        }
        
        // Clean up
        gpu->deallocate_memory_slab(mem);
        std::cout << "Successfully deallocated GPU memory\n";
        
        std::cout << "\nGPU substrate test completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "GPU substrate test failed: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}