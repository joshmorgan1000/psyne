/**
 * @file vulkan_buffer.hpp
 * @brief Vulkan buffer implementation for cross-platform GPU support
 *
 * Provides Vulkan-based GPU buffer management for AMD, NVIDIA, and Intel GPUs
 * with support for compute operations and memory sharing.
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#pragma once

#include "../gpu_buffer.hpp"
#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace psyne {
namespace gpu {
namespace vulkan {

/**
 * @brief Vulkan-specific GPU buffer implementation
 */
class VulkanBuffer : public GPUBuffer {
public:
    VulkanBuffer(VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
                 size_t size, BufferUsage usage, MemoryAccess access);
    ~VulkanBuffer() override;

    // GPUBuffer interface
    size_t size() const override {
        return size_;
    }
    BufferUsage usage() const override {
        return usage_;
    }
    MemoryAccess access() const override {
        return access_;
    }
    void *map() override;
    void unmap() override;
    void flush() override;
    void *native_handle() override {
        return buffer_;
    }
    bool is_mapped() const override {
        return mapped_ptr_ != nullptr;
    }
    void upload(const void *data, size_t size, size_t offset = 0) override;
    void download(void *data, size_t size, size_t offset = 0) override;

    /**
     * @brief Get Vulkan buffer handle
     */
    VkBuffer vulkan_buffer() const {
        return buffer_;
    }

    /**
     * @brief Get device memory handle
     */
    VkDeviceMemory device_memory() const {
        return memory_;
    }

private:
    VkDevice device_;
    VkBuffer buffer_;
    VkDeviceMemory memory_;
    size_t size_;
    BufferUsage usage_;
    MemoryAccess access_;
    void *mapped_ptr_;
};

/**
 * @brief Vulkan GPU buffer factory
 */
class VulkanBufferFactory : public GPUBufferFactory {
public:
    VulkanBufferFactory(VkDevice device, VkPhysicalDevice physical_device,
                        uint32_t queue_family_index);
    ~VulkanBufferFactory() override;

    std::unique_ptr<GPUBuffer>
    create_buffer(size_t size, BufferUsage usage = BufferUsage::Dynamic,
                  MemoryAccess access = MemoryAccess::Shared) override;

    GPUBackend backend() const override {
        return GPUBackend::Vulkan;
    }
    bool supports_unified_memory() const override;
    size_t max_buffer_size() const override;

private:
    VkDevice device_;
    VkPhysicalDevice physical_device_;
    uint32_t queue_family_index_;
    VkPhysicalDeviceMemoryProperties memory_properties_;

    uint32_t find_memory_type(uint32_t type_filter,
                              VkMemoryPropertyFlags properties);
    VkBufferUsageFlags get_buffer_usage_flags(BufferUsage usage,
                                              MemoryAccess access);
    VkMemoryPropertyFlags get_memory_property_flags(MemoryAccess access);
};

/**
 * @brief Vulkan GPU context
 */
class VulkanContext : public GPUContext {
public:
    VulkanContext();
    ~VulkanContext() override;

    GPUBackend backend() const override {
        return GPUBackend::Vulkan;
    }
    std::unique_ptr<GPUBufferFactory> create_buffer_factory() override;
    std::string device_name() const override;
    size_t total_memory() const override;
    size_t available_memory() const override;
    bool is_unified_memory() const override;
    void synchronize() override;

    /**
     * @brief Get Vulkan instance
     */
    VkInstance instance() const {
        return instance_;
    }

    /**
     * @brief Get physical device
     */
    VkPhysicalDevice physical_device() const {
        return physical_device_;
    }

    /**
     * @brief Get logical device
     */
    VkDevice device() const {
        return device_;
    }

    /**
     * @brief Get compute queue
     */
    VkQueue compute_queue() const {
        return compute_queue_;
    }

    /**
     * @brief Get compute queue family index
     */
    uint32_t compute_queue_family() const {
        return compute_queue_family_;
    }

    /**
     * @brief Create compute pipeline
     */
    VkPipeline create_compute_pipeline(VkShaderModule shader_module,
                                       VkPipelineLayout pipeline_layout);

    /**
     * @brief Load shader from SPIR-V
     */
    VkShaderModule load_shader_module(const std::vector<uint32_t> &spirv_code);

private:
    VkInstance instance_;
    VkPhysicalDevice physical_device_;
    VkDevice device_;
    VkQueue compute_queue_;
    VkCommandPool command_pool_;
    uint32_t compute_queue_family_;

    // Device properties
    VkPhysicalDeviceProperties device_properties_;
    VkPhysicalDeviceMemoryProperties memory_properties_;
    VkPhysicalDeviceFeatures device_features_;

    bool initialize();
    bool create_instance();
    bool select_physical_device();
    bool create_logical_device();
    uint32_t find_compute_queue_family();

    // Validation layers
#ifdef DEBUG
    VkDebugUtilsMessengerEXT debug_messenger_;
    static VKAPI_ATTR VkBool32 VKAPI_CALL
    debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                   VkDebugUtilsMessageTypeFlagsEXT type,
                   const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
                   void *user_data);
#endif
};

/**
 * @brief RAII wrapper for Vulkan command buffer
 */
class VulkanCommandBuffer {
public:
    VulkanCommandBuffer(VkDevice device, VkCommandPool pool);
    ~VulkanCommandBuffer();

    VkCommandBuffer get() const {
        return buffer_;
    }

    void begin();
    void end();
    void submit(VkQueue queue, VkFence fence = VK_NULL_HANDLE);

    // Compute operations
    void bind_pipeline(VkPipeline pipeline);
    void bind_descriptor_set(VkPipelineLayout layout, VkDescriptorSet set);
    void dispatch(uint32_t x, uint32_t y, uint32_t z);

    // Buffer operations
    void copy_buffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);
    void fill_buffer(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                     uint32_t data);

    // Synchronization
    void pipeline_barrier(VkPipelineStageFlags src_stage,
                          VkPipelineStageFlags dst_stage,
                          VkBufferMemoryBarrier *barrier);

private:
    VkDevice device_;
    VkCommandPool pool_;
    VkCommandBuffer buffer_;
};

/**
 * @brief Vulkan compute shader wrapper
 */
class VulkanComputeShader {
public:
    VulkanComputeShader(VkDevice device);
    ~VulkanComputeShader();

    /**
     * @brief Load shader from SPIR-V code
     */
    bool load_spirv(const std::vector<uint32_t> &spirv_code);

    /**
     * @brief Create pipeline with descriptor set layout
     */
    bool
    create_pipeline(const std::vector<VkDescriptorSetLayoutBinding> &bindings);

    /**
     * @brief Bind buffer to descriptor set
     */
    void bind_buffer(uint32_t binding, VkBuffer buffer, VkDeviceSize size);

    /**
     * @brief Execute compute shader
     */
    void execute(VulkanCommandBuffer &cmd, uint32_t x, uint32_t y, uint32_t z);

    VkPipeline pipeline() const {
        return pipeline_;
    }
    VkPipelineLayout pipeline_layout() const {
        return pipeline_layout_;
    }
    VkDescriptorSet descriptor_set() const {
        return descriptor_set_;
    }

private:
    VkDevice device_;
    VkShaderModule shader_module_;
    VkPipeline pipeline_;
    VkPipelineLayout pipeline_layout_;
    VkDescriptorSetLayout descriptor_set_layout_;
    VkDescriptorPool descriptor_pool_;
    VkDescriptorSet descriptor_set_;

    void cleanup();
};

/**
 * @brief Vulkan memory allocator for optimal memory management
 */
class VulkanMemoryAllocator {
public:
    VulkanMemoryAllocator(VkDevice device, VkPhysicalDevice physical_device);
    ~VulkanMemoryAllocator();

    struct Allocation {
        VkDeviceMemory memory;
        VkDeviceSize offset;
        VkDeviceSize size;
    };

    /**
     * @brief Allocate memory for buffer
     */
    Allocation allocate(VkMemoryRequirements requirements,
                        VkMemoryPropertyFlags properties);

    /**
     * @brief Free allocated memory
     */
    void free(const Allocation &allocation);

    /**
     * @brief Get statistics
     */
    struct Stats {
        size_t total_allocated;
        size_t total_free;
        size_t allocation_count;
        size_t memory_type_count;
    };

    Stats get_stats() const;

private:
    VkDevice device_;
    VkPhysicalDevice physical_device_;
    VkPhysicalDeviceMemoryProperties memory_properties_;

    // Memory pools per memory type
    struct MemoryPool {
        VkDeviceMemory memory;
        VkDeviceSize size;
        VkDeviceSize used;
        std::vector<Allocation> allocations;
    };

    std::vector<MemoryPool> pools_;
    std::mutex mutex_;

    uint32_t find_memory_type(uint32_t type_filter,
                              VkMemoryPropertyFlags properties);
};

} // namespace vulkan
} // namespace gpu
} // namespace psyne