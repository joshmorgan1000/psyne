/**
 * @file gpu_vulkan_test.cpp
 * @brief Test Vulkan GPU substrate specifically
 */

#include <psyne/config_detect.hpp>
#include <iostream>

int main() {
    std::cout << "=== Vulkan GPU Substrate Test ===\n\n";
    
#ifdef PSYNE_VULKAN_ENABLED
    std::cout << "Vulkan support is ENABLED\n";
    
    // Try to create a simple Vulkan substrate directly
    #include <psyne/channel/substrate/gpu_vulkan.hpp>
    
    try {
        psyne::substrate::GPUVulkan vulkan_substrate(0);
        std::cout << "Vulkan substrate created successfully!\n";
        
        // Try to allocate some memory
        void* mem = vulkan_substrate.allocate_memory_slab(1024 * 1024); // 1MB
        if (mem) {
            std::cout << "Successfully allocated 1MB of Vulkan GPU memory\n";
            
            // Write a test pattern
            unsigned char* bytes = static_cast<unsigned char*>(mem);
            for (int i = 0; i < 256; ++i) {
                bytes[i] = static_cast<unsigned char>(i);
            }
            
            // Verify
            bool success = true;
            for (int i = 0; i < 256; ++i) {
                if (bytes[i] != static_cast<unsigned char>(i)) {
                    success = false;
                    break;
                }
            }
            
            if (success) {
                std::cout << "Memory read/write test PASSED\n";
            } else {
                std::cout << "Memory read/write test FAILED\n";
            }
            
            vulkan_substrate.deallocate_memory_slab(mem);
            std::cout << "Memory deallocated successfully\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Vulkan test failed: " << e.what() << "\n";
        return 1;
    }
#else
    std::cout << "Vulkan support is DISABLED\n";
#endif
    
    return 0;
}