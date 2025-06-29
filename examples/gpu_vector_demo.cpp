#include <psyne/psyne.hpp>
#include <psyne/gpu/gpu_buffer.hpp>
#include <psyne/gpu/gpu_message.hpp>
#include <iostream>

using namespace psyne;
using namespace psyne::gpu;

int main() {
    std::cout << "GPU Vector Demo\n";
    std::cout << "===============\n\n";
    
    try {
        // Create a GPU context (Metal on macOS)
        auto gpu_context = create_gpu_context(GPUBackend::Metal);
        if (!gpu_context) {
            std::cout << "No compatible GPU found. This demo requires Metal support.\n";
            std::cout << "Make sure you're running on macOS with Metal support.\n";
            return 1;
        }
        
        std::cout << "GPU Context created successfully\n";
        std::cout << "Backend: " << (gpu_context->backend() == GPUBackend::Metal ? "Metal" : "Unknown") << "\n\n";
        
        // Create a channel for GPU-aware messages
        auto channel = create_channel("memory://gpu_demo", 
                                      1024 * 1024, 
                                      ChannelMode::SPSC, 
                                      ChannelType::SingleType);
        
        // Create a GPU-aware float vector
        GPUFloatVector gpu_vector(*channel);
        if (!gpu_vector.is_valid()) {
            std::cerr << "Failed to create GPU vector\n";
            return 1;
        }
        
        // Fill with test data
        std::cout << "1. Creating CPU data...\n";
        gpu_vector = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        
        std::cout << "   Vector size: " << gpu_vector.size() << "\n";
        std::cout << "   CPU data: ";
        for (float val : gpu_vector) {
            std::cout << val << " ";
        }
        std::cout << "\n\n";
        
        // Create GPU buffer
        std::cout << "2. Transferring to GPU...\n";
        auto gpu_buffer = gpu_vector.to_gpu_buffer(*gpu_context);
        if (!gpu_buffer) {
            std::cerr << "Failed to create GPU buffer\n";
            return 1;
        }
        
        std::cout << "   GPU buffer created: " << gpu_buffer->size() << " bytes\n";
        std::cout << "   Memory access: " << (gpu_buffer->access() == MemoryAccess::Shared ? "Shared" : "Other") << "\n";
        std::cout << "   Is on GPU: " << (gpu_vector.is_on_gpu() ? "Yes" : "No") << "\n\n";
        
        // Demonstrate unified memory access
        std::cout << "3. Unified memory demonstration...\n";
        if (gpu_buffer->access() == MemoryAccess::Shared) {
            std::cout << "   This GPU buffer uses unified memory!\n";
            std::cout << "   CPU and GPU can access the same memory location.\n";
            std::cout << "   Perfect for Apple Silicon's unified memory architecture.\n\n";
        }
        
        // Perform GPU operation (scaling)
        std::cout << "4. GPU compute operation (scaling by 2.0)...\n";
        gpu_vector.gpu_scale(*gpu_context, 2.0f);
        
        std::cout << "   Scaled data: ";
        for (float val : gpu_vector) {
            std::cout << val << " ";
        }
        std::cout << "\n\n";
        
        // Send the GPU-aware message
        std::cout << "5. Sending GPU-aware message...\n";
        channel->send(gpu_vector);
        
        // Receive and process
        auto received = channel->receive<GPUFloatVector>();
        if (received) {
            std::cout << "   Received GPU vector with " << received->size() << " elements\n";
            std::cout << "   Data: ";
            for (float val : *received) {
                std::cout << val << " ";
            }
            std::cout << "\n\n";
            
            // Demonstrate Eigen integration
            std::cout << "6. Eigen integration...\n";
            auto eigen_view = received->as_eigen();
            std::cout << "   L2 norm: " << eigen_view.norm() << "\n";
            std::cout << "   Sum: " << eigen_view.sum() << "\n";
            std::cout << "   Mean: " << eigen_view.mean() << "\n\n";
        }
        
        // Demonstrate GPU buffer factory
        std::cout << "7. Direct GPU buffer creation...\n";
        auto factory = gpu_context->create_buffer_factory();
        if (factory) {
            auto direct_buffer = factory->create_buffer(
                1024, 
                BufferUsage::Dynamic, 
                MemoryAccess::Shared
            );
            
            if (direct_buffer) {
                std::cout << "   Created direct GPU buffer: " << direct_buffer->size() << " bytes\n";
                
                // Map and write data
                float* buffer_ptr = static_cast<float*>(direct_buffer->map());
                if (buffer_ptr) {
                    for (int i = 0; i < 10; ++i) {
                        buffer_ptr[i] = i * 0.5f;
                    }
                    direct_buffer->unmap();
                    direct_buffer->flush();
                    
                    std::cout << "   Wrote test data to GPU buffer\n";
                }
            }
        }
        
        std::cout << "\nGPU demo completed successfully!\n";
        std::cout << "This demonstrates zero-copy GPU integration on unified memory systems.\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}