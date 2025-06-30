/**
 * @brief Simple CUDA test to verify the implementation works
 */

#include <iostream>
#include <psyne/psyne.hpp>

using namespace psyne;

int main() {
    std::cout << "Testing CUDA support in Psyne..." << std::endl;

    // Check available GPU backends
    auto backends = gpu::detect_gpu_backends();
    std::cout << "Available GPU backends: " << backends.size() << std::endl;
    for (auto backend : backends) {
        std::cout << "  - " << gpu::gpu_backend_name(backend) << std::endl;
    }

    // Try to create CUDA context
    if (std::find(backends.begin(), backends.end(), gpu::GPUBackend::CUDA) !=
        backends.end()) {
        std::cout << "\nCUDA backend is available! ✓" << std::endl;

        auto gpu_context = gpu::create_gpu_context(gpu::GPUBackend::CUDA);
        if (gpu_context) {
            std::cout << "✓ CUDA context created successfully!" << std::endl;
            // Note: Can't call methods on GPUContext as it's only forward
            // declared in public API This is intentional - internal GPU details
            // are hidden from public API
        } else {
            std::cout << "✗ Failed to create CUDA context" << std::endl;
        }
    } else {
        std::cout << "CUDA not available on this system." << std::endl;
    }

    std::cout << "\nCUDA integration test completed!" << std::endl;
    return 0;
}