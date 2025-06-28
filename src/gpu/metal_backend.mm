#include "metal_backend.hpp"
#include <iostream>
#include <unordered_map>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

namespace psyne {
namespace gpu {

// MetalBuffer implementation
MetalBuffer::MetalBuffer(id<MTLBuffer> buffer, BufferUsage usage, MemoryAccess access)
    : buffer_(buffer)
    , usage_(usage)
    , access_(access)
    , mapped_(false)
    , mapped_ptr_(nullptr) {
    [buffer_ retain];
}

MetalBuffer::~MetalBuffer() {
    if (mapped_) {
        unmap();
    }
    [buffer_ release];
}

size_t MetalBuffer::size() const {
    return [buffer_ length];
}

void* MetalBuffer::map() {
    if (mapped_) {
        return mapped_ptr_;
    }
    
    if (access_ == MemoryAccess::GPUOnly) {
        return nullptr;  // Cannot map GPU-only memory
    }
    
    mapped_ptr_ = [buffer_ contents];
    mapped_ = true;
    return mapped_ptr_;
}

const void* MetalBuffer::map() const {
    if (!mapped_) {
        return const_cast<MetalBuffer*>(this)->map();
    }
    return mapped_ptr_;
}

void MetalBuffer::unmap() {
    if (mapped_) {
        mapped_ = false;
        mapped_ptr_ = nullptr;
        // Note: Metal buffers with shared storage don't need explicit unmapping
    }
}

void MetalBuffer::flush() {
    // On unified memory systems (Apple Silicon), this is typically a no-op
    // But we could add explicit synchronization if needed
}

void MetalBuffer::invalidate() {
    // On unified memory systems, this is typically a no-op
}

void* MetalBuffer::native_handle() {
    return (__bridge void*)buffer_;
}

const void* MetalBuffer::native_handle() const {
    return (__bridge const void*)buffer_;
}

// MetalBufferFactory implementation
MetalBufferFactory::MetalBufferFactory(id<MTLDevice> device)
    : device_(device) {
    [device_ retain];
}

MetalBufferFactory::~MetalBufferFactory() {
    [device_ release];
}

std::unique_ptr<GPUBuffer> MetalBufferFactory::create_buffer(
    size_t size, BufferUsage usage, MemoryAccess access) {
    
    MTLResourceOptions options = 0;
    
    switch (access) {
        case MemoryAccess::GPUOnly:
            options = MTLResourceStorageModePrivate;
            break;
        case MemoryAccess::Shared:
            options = MTLResourceStorageModeShared;
            break;
        case MemoryAccess::Staging:
            options = MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
            break;
    }
    
    id<MTLBuffer> buffer = [device_ newBufferWithLength:size options:options];
    if (!buffer) {
        return nullptr;
    }
    
    return std::make_unique<MetalBuffer>(buffer, usage, access);
}

bool MetalBufferFactory::is_available() const {
    return device_ != nil;
}

// MetalContext implementation
MetalContext::MetalContext()
    : device_(nil)
    , command_queue_(nil)
    , library_(nil) {
    initialize();
}

MetalContext::~MetalContext() {
    [library_ release];
    [command_queue_ release];
    [device_ release];
}

bool MetalContext::initialize() {
    // Get default Metal device
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
        std::cerr << "Metal: No compatible GPU found\n";
        return false;
    }
    [device_ retain];
    
    // Create command queue
    command_queue_ = [device_ newCommandQueue];
    if (!command_queue_) {
        std::cerr << "Metal: Failed to create command queue\n";
        return false;
    }
    
    // Create default library (for built-in shaders)
    library_ = [device_ newDefaultLibrary];
    // Note: library_ might be nil if no default shaders, that's okay
    
    return true;
}

std::unique_ptr<GPUBufferFactory> MetalContext::create_buffer_factory() {
    if (!device_) {
        return nullptr;
    }
    return std::make_unique<MetalBufferFactory>(device_);
}

void MetalContext::dispatch_compute(
    const void* shader_data, size_t shader_size,
    GPUBuffer& input, GPUBuffer& output,
    size_t workgroup_x, size_t workgroup_y, size_t workgroup_z) {
    
    // This is a simplified implementation
    // In practice, you'd compile the shader and create a compute pipeline
    
    id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    
    // Get Metal buffers
    auto* input_metal = static_cast<MetalBuffer*>(&input);
    auto* output_metal = static_cast<MetalBuffer*>(&output);
    
    // Set buffers (this is pseudo-code - real implementation would set up pipeline)
    [encoder setBuffer:input_metal->metal_buffer() offset:0 atIndex:0];
    [encoder setBuffer:output_metal->metal_buffer() offset:0 atIndex:1];
    
    // Dispatch (this needs a real compute pipeline state)
    // MTLSize gridSize = MTLSizeMake(workgroup_x, workgroup_y, workgroup_z);
    // [encoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

void MetalContext::sync() {
    // Wait for all pending operations to complete
    // In a real implementation, you'd track active command buffers
}

id<MTLComputePipelineState> MetalContext::get_compute_pipeline(const std::string& shader_source) {
    auto it = pipeline_cache_.find(shader_source);
    if (it != pipeline_cache_.end()) {
        return it->second;
    }
    
    // Compile shader from source
    NSString* source = [NSString stringWithUTF8String:shader_source.c_str()];
    NSError* error = nil;
    
    id<MTLLibrary> library = [device_ newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        std::cerr << "Metal: Failed to compile shader: " << [[error localizedDescription] UTF8String] << std::endl;
        return nil;
    }
    
    id<MTLFunction> function = [library newFunctionWithName:@"compute_main"];
    if (!function) {
        std::cerr << "Metal: Function 'compute_main' not found in shader\n";
        [library release];
        return nil;
    }
    
    id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        std::cerr << "Metal: Failed to create pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
        [function release];
        [library release];
        return nil;
    }
    
    // Cache the pipeline
    pipeline_cache_[shader_source] = pipeline;
    
    [function release];
    [library release];
    
    return pipeline;
}

} // namespace gpu
} // namespace psyne

// Factory function implementation
namespace psyne {
namespace gpu {

std::unique_ptr<GPUContext> create_metal_context() {
    auto context = std::make_unique<MetalContext>();
    if (!context->device()) {
        return nullptr;
    }
    return std::unique_ptr<GPUContext>(context.release());
}

std::unique_ptr<GPUContext> create_gpu_context(GPUBackend preferred) {
    switch (preferred) {
        case GPUBackend::Metal:
            return create_metal_context();
        default:
            // Try Metal first on macOS
            auto metal_context = create_metal_context();
            if (metal_context) {
                return metal_context;
            }
            break;
    }
    
    return nullptr;
}

} // namespace gpu
} // namespace psyne