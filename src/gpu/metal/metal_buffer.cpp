/**
 * @file metal_buffer.cpp
 * @brief Apple Metal buffer implementation
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#ifdef __APPLE__

#include <psyne/gpu/metal/metal_buffer.hpp>
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace psyne {
namespace gpu {
namespace metal {

// MetalBuffer implementation

MetalBuffer::MetalBuffer(MTL::Buffer* buffer, BufferUsage usage, MemoryAccess access)
    : buffer_(buffer)
    , usage_(usage)
    , access_(access)
    , mapped_ptr_(nullptr)
    , is_mapped_(false) {
    
    if (!buffer_) {
        throw std::runtime_error("Invalid Metal buffer");
    }
    
    buffer_->retain();
}

MetalBuffer::~MetalBuffer() {
    if (is_mapped_) {
        unmap();
    }
    
    if (buffer_) {
        buffer_->release();
    }
}

size_t MetalBuffer::size() const {
    return buffer_->length();
}

BufferUsage MetalBuffer::usage() const {
    return usage_;
}

MemoryAccess MetalBuffer::access() const {
    return access_;
}

void* MetalBuffer::map() {
    if (is_mapped_) {
        return mapped_ptr_;
    }
    
    // For Metal with unified memory, contents() gives direct CPU access
    mapped_ptr_ = buffer_->contents();
    is_mapped_ = true;
    
    return mapped_ptr_;
}

void MetalBuffer::unmap() {
    if (!is_mapped_) {
        return;
    }
    
    // On unified memory systems, we might need to ensure CPU writes are visible to GPU
    if (access_ == MemoryAccess::Shared || access_ == MemoryAccess::Managed) {
        // For shared/managed storage mode, CPU and GPU automatically stay coherent
        // No explicit action needed on Apple Silicon
    }
    
    mapped_ptr_ = nullptr;
    is_mapped_ = false;
}

void MetalBuffer::flush() {
    // On Apple Silicon with unified memory, no explicit flush needed
    // CPU and GPU share the same memory and cache hierarchy
    
    // For discrete GPUs (older Macs), we might need:
    // buffer_->didModifyRange(NS::Range::Make(0, buffer_->length()));
}

void* MetalBuffer::native_handle() {
    return buffer_;
}

bool MetalBuffer::is_mapped() const {
    return is_mapped_;
}

void MetalBuffer::upload(const void* data, size_t size, size_t offset) {
    if (offset + size > buffer_->length()) {
        throw std::out_of_range("Upload exceeds buffer size");
    }
    
    void* dst = map();
    if (!dst) {
        throw std::runtime_error("Failed to map buffer for upload");
    }
    
    std::memcpy(static_cast<uint8_t*>(dst) + offset, data, size);
    unmap();
    flush();
}

void MetalBuffer::download(void* data, size_t size, size_t offset) {
    if (offset + size > buffer_->length()) {
        throw std::out_of_range("Download exceeds buffer size");
    }
    
    void* src = map();
    if (!src) {
        throw std::runtime_error("Failed to map buffer for download");
    }
    
    std::memcpy(data, static_cast<uint8_t*>(src) + offset, size);
    unmap();
}

// MetalBufferFactory implementation

MetalBufferFactory::MetalBufferFactory(MTL::Device* device)
    : device_(device) {
    
    if (!device_) {
        throw std::runtime_error("Invalid Metal device");
    }
    
    device_->retain();
}

MetalBufferFactory::~MetalBufferFactory() {
    if (device_) {
        device_->release();
    }
}

std::unique_ptr<GPUBuffer> MetalBufferFactory::create_buffer(
    size_t size, BufferUsage usage, MemoryAccess access) {
    
    if (size == 0) {
        throw std::invalid_argument("Buffer size cannot be zero");
    }
    
    if (size > max_buffer_size()) {
        throw std::invalid_argument("Buffer size exceeds maximum");
    }
    
    MTL::ResourceOptions options = get_resource_options(usage, access);
    
    // Create Metal buffer
    MTL::Buffer* metal_buffer = device_->newBuffer(size, options);
    if (!metal_buffer) {
        throw std::runtime_error("Failed to create Metal buffer");
    }
    
    try {
        return std::make_unique<MetalBuffer>(metal_buffer, usage, access);
    } catch (...) {
        metal_buffer->release();
        throw;
    }
}

size_t MetalBufferFactory::max_buffer_size() const {
    // Metal supports very large buffers, but let's be reasonable
    // This can be queried from device if needed
    return 1ULL << 32; // 4GB
}

MTL::ResourceOptions MetalBufferFactory::get_resource_options(
    BufferUsage usage, MemoryAccess access) const {
    
    MTL::ResourceOptions options = 0;
    
    // Storage mode based on access pattern
    switch (access) {
        case MemoryAccess::DeviceOnly:
            options |= MTL::ResourceStorageModePrivate;
            break;
            
        case MemoryAccess::HostOnly:
        case MemoryAccess::Shared:
            // Shared storage mode for unified memory architecture
            options |= MTL::ResourceStorageModeShared;
            break;
            
        case MemoryAccess::Managed:
            // Managed storage mode for automatic CPU/GPU synchronization
            options |= MTL::ResourceStorageModeManaged;
            break;
    }
    
    // CPU cache mode
    if (access == MemoryAccess::Shared || access == MemoryAccess::HostOnly) {
        options |= MTL::ResourceCPUCacheModeDefaultCache;
    }
    
    // Hazard tracking for safety
    options |= MTL::ResourceHazardTrackingModeDefault;
    
    return options;
}

// MetalContext implementation

MetalContext::MetalContext()
    : device_(nullptr)
    , command_queue_(nullptr)
    , default_library_(nullptr) {
    
    if (!initialize()) {
        throw std::runtime_error("Failed to initialize Metal context");
    }
}

MetalContext::~MetalContext() {
    if (default_library_) {
        default_library_->release();
    }
    
    if (command_queue_) {
        command_queue_->release();
    }
    
    if (device_) {
        device_->release();
    }
}

bool MetalContext::initialize() {
    // Get default Metal device
    device_ = MTL::CreateSystemDefaultDevice();
    if (!device_) {
        std::cerr << "No Metal device found" << std::endl;
        return false;
    }
    
    // Create command queue
    command_queue_ = device_->newCommandQueue();
    if (!command_queue_) {
        std::cerr << "Failed to create Metal command queue" << std::endl;
        return false;
    }
    
    // Load default library (for compute shaders)
    default_library_ = device_->newDefaultLibrary();
    // It's okay if this is null - we might not have any Metal shaders yet
    
    std::cout << "Metal context initialized: " << device_name() << std::endl;
    std::cout << "Unified memory: " << (is_unified_memory() ? "Yes" : "No") << std::endl;
    
    return true;
}

std::unique_ptr<GPUBufferFactory> MetalContext::create_buffer_factory() {
    return std::make_unique<MetalBufferFactory>(device_);
}

std::string MetalContext::device_name() const {
    if (!device_) {
        return "Unknown";
    }
    
    NS::String* name = device_->name();
    return name ? name->utf8String() : "Unknown Metal Device";
}

size_t MetalContext::total_memory() const {
    // Note: recommendedMaxWorkingSetSize gives a recommended memory limit
    // For actual total memory, we'd need to query system info
    return device_->recommendedMaxWorkingSetSize();
}

size_t MetalContext::available_memory() const {
    // Metal doesn't directly expose available memory
    // This would require system-level queries
    return total_memory(); // Approximation
}

bool MetalContext::is_unified_memory() const {
    // Check if this is an Apple Silicon Mac with unified memory
    return device_->hasUnifiedMemory();
}

void MetalContext::synchronize() {
    // Create a command buffer and wait for it to complete
    // This ensures all GPU work is done
    MTL::CommandBuffer* cmd_buffer = command_queue_->commandBuffer();
    if (cmd_buffer) {
        cmd_buffer->commit();
        cmd_buffer->waitUntilCompleted();
        cmd_buffer->release();
    }
}

MTL::ComputePipelineState* MetalContext::create_compute_pipeline(
    const std::string& function_name) {
    
    if (!default_library_) {
        throw std::runtime_error("No Metal library loaded");
    }
    
    NS::String* fn_name = NS::String::string(function_name.c_str(), 
                                             NS::StringEncoding::UTF8StringEncoding);
    MTL::Function* function = default_library_->newFunction(fn_name);
    fn_name->release();
    
    if (!function) {
        throw std::runtime_error("Metal function not found: " + function_name);
    }
    
    NS::Error* error = nullptr;
    MTL::ComputePipelineState* pipeline = device_->newComputePipelineState(function, &error);
    function->release();
    
    if (!pipeline || error) {
        std::string error_msg = "Failed to create compute pipeline";
        if (error) {
            error_msg += ": ";
            error_msg += error->localizedDescription()->utf8String();
            error->release();
        }
        throw std::runtime_error(error_msg);
    }
    
    return pipeline;
}

// MetalCommandBuffer implementation

MetalCommandBuffer::MetalCommandBuffer(MTL::CommandQueue* queue)
    : buffer_(nullptr)
    , encoder_(nullptr) {
    
    if (!queue) {
        throw std::invalid_argument("Invalid command queue");
    }
    
    buffer_ = queue->commandBuffer();
    if (!buffer_) {
        throw std::runtime_error("Failed to create command buffer");
    }
    
    buffer_->retain();
}

MetalCommandBuffer::~MetalCommandBuffer() {
    if (encoder_) {
        encoder_->endEncoding();
        encoder_->release();
    }
    
    if (buffer_) {
        buffer_->release();
    }
}

MTL::ComputeCommandEncoder* MetalCommandBuffer::compute_encoder() {
    if (!encoder_) {
        encoder_ = buffer_->computeCommandEncoder();
        if (!encoder_) {
            throw std::runtime_error("Failed to create compute encoder");
        }
        encoder_->retain();
    }
    
    return encoder_;
}

void MetalCommandBuffer::commit() {
    if (encoder_) {
        encoder_->endEncoding();
        encoder_->release();
        encoder_ = nullptr;
    }
    
    buffer_->commit();
}

void MetalCommandBuffer::wait_until_completed() {
    buffer_->waitUntilCompleted();
}

} // namespace metal
} // namespace gpu
} // namespace psyne

#endif // __APPLE__