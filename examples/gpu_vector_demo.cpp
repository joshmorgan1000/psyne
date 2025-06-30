#include <iostream>
#include <psyne/psyne.hpp>
#include <Eigen/Core>

using namespace psyne;
using namespace psyne::gpu;

int main() {
    std::cout << "GPU Vector Demo\n";
    std::cout << "===============\n\n";

    try {
        // Create a GPU context (try CUDA first, then fallback)
        std::unique_ptr<GPUContext> gpu_context;
        
#ifdef PSYNE_GPU_SUPPORT
        try {
            gpu_context = create_gpu_context(GPUBackend::CUDA);
        } catch (...) {
            gpu_context = nullptr;
        }
#endif

        if (gpu_context) {
            std::cout << "GPU Context created successfully\n";
            std::cout << "Backend: " << gpu_backend_name(gpu_context->backend()) << "\n\n";
        } else {
            std::cout << "GPU context not available, running CPU simulation\n\n";
        }

        // Create a channel for GPU-aware messages
        auto channel =
            create_channel("memory://gpu_demo", 1024 * 1024, ChannelMode::SPSC,
                           ChannelType::SingleType);

        // Create a float vector (with GPU capabilities)
        FloatVector gpu_vector(*channel);

        // Fill with test data
        std::cout << "1. Creating unified data views...\n";
        gpu_vector = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

        std::cout << "   Vector size: " << gpu_vector.size() << "\n";
        
        // Demonstrate the unified view philosophy
        std::cout << "   C++ range-based loop: ";
        for (float val : gpu_vector) {
            std::cout << val << " ";
        }
        std::cout << "\n";
        
        // Same data via C++ pointer access
        std::cout << "   C++ pointer access: ";
        float* float_ptr = reinterpret_cast<float*>(gpu_vector.data());
        for (size_t i = 0; i < gpu_vector.size(); ++i) {
            std::cout << float_ptr[i] << " ";
        }
        std::cout << "\n";
        
        // Same data via Eigen view (zero-copy)
        auto eigen_view = Eigen::Map<Eigen::VectorXf>(
            reinterpret_cast<float*>(gpu_vector.data()), gpu_vector.size());
        std::cout << "   Eigen view access: ";
        for (int i = 0; i < eigen_view.size(); ++i) {
            std::cout << eigen_view[i] << " ";
        }
        std::cout << "\n";
        std::cout << "   All views point to: " << (void*)gpu_vector.data() << "\n\n";

        // GPU buffer simulation
        std::cout << "2. GPU buffer simulation...\n";
        std::cout << "   Vector data size: " << (gpu_vector.size() * sizeof(float))
                  << " bytes\n";
        std::cout << "   Memory access: CPU-based (GPU simulation)\n";
        std::cout << "   Is on GPU: No (CPU fallback)\n\n";

        // Demonstrate unified memory access
        std::cout << "3. Unified memory demonstration...\n";
        std::cout << "   This demo shows unified memory concepts.\n";
        std::cout << "   CPU and GPU can access the same memory location.\n";
        std::cout << "   Perfect for unified memory architectures.\n\n";

        // Perform scaling operation showing unified access
        std::cout << "4. Unified compute operation (scaling by 2.0)...\n";
        
        // Option 1: C++ loop
        std::cout << "   Using C++ loop...\n";
        for (size_t i = 0; i < gpu_vector.size(); ++i) {
            gpu_vector[i] *= 2.0f;
        }
        
        // Option 2: Eigen operations on same data (zero-copy)
        std::cout << "   Using Eigen operations on same memory...\n";
        auto eigen_compute_view = Eigen::Map<Eigen::VectorXf>(
            reinterpret_cast<float*>(gpu_vector.data()), gpu_vector.size());
        // This could be: eigen_compute_view = eigen_compute_view.array() * 2.0f;
        // But we already scaled above, so let's do a different operation
        float mean_val = eigen_compute_view.mean();
        std::cout << "   Eigen computed mean: " << mean_val << "\n";
        
        // This memory will also be GPU-accessible in a real CUDA kernel
        std::cout << "   Memory layout ready for GPU kernels\n";

        std::cout << "   Scaled data: ";
        for (float val : gpu_vector) {
            std::cout << val << " ";
        }
        std::cout << "\n\n";

        // Send the GPU-aware message
        std::cout << "5. Sending GPU-aware message...\n";
        channel->send(gpu_vector);

        // Receive and process
        auto received = channel->receive<FloatVector>();
        if (received) {
            std::cout << "   Received GPU vector with " << received->size()
                      << " elements\n";
            std::cout << "   Data: ";
            for (float val : *received) {
                std::cout << val << " ";
            }
            std::cout << "\n\n";

            // Demonstrate Eigen integration - unified view philosophy
            std::cout << "6. Eigen integration (unified view)...\n";
            
            // Create Eigen view of the same data - zero copy!
            auto eigen_view = Eigen::Map<Eigen::VectorXf>(
                reinterpret_cast<float*>(received->data()), received->size());
            
            std::cout << "   L2 norm: " << eigen_view.norm() << "\n";
            std::cout << "   Sum: " << eigen_view.sum() << "\n";
            std::cout << "   Mean: " << eigen_view.mean() << "\n";
            std::cout << "   Data pointer (C++): " << (void*)received->data() << "\n";
            std::cout << "   Data pointer (Eigen): " << (void*)eigen_view.data() << "\n";
            std::cout << "   → Same memory location = zero-copy!\n\n";
        }

        // Demonstrate direct buffer concept
        std::cout << "7. Direct buffer creation...\n";
        std::vector<float> direct_buffer(256);
        std::cout << "   Created direct buffer: " << (direct_buffer.size() * sizeof(float)) << " bytes\n";
        
        // Fill with test data
        for (int i = 0; i < 10; ++i) {
            direct_buffer[i] = i * 0.5f;
        }
        std::cout << "   Wrote test data to buffer\n";

        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "UNIFIED VIEW PHILOSOPHY DEMONSTRATED\n";
        std::cout << std::string(60, '=') << "\n";
        std::cout << "✓ Same memory accessible via:\n";
        std::cout << "  • C++ pointers (float*)\n";
        std::cout << "  • C++ range-based loops\n";
        std::cout << "  • Eigen mathematical operations\n";
        std::cout << "  • GPU shader memory (when available)\n";
        std::cout << "  • Zero-copy message passing\n\n";
        std::cout << "✓ All views operate on identical memory location\n";
        std::cout << "✓ No data copying between view types\n";
        std::cout << "✓ Maximum performance with unified access patterns\n\n";
        std::cout << "This demonstrates psyne's core philosophy:\n";
        std::cout << "ONE MEMORY LAYOUT → MULTIPLE ACCESS PATTERNS\n";

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}