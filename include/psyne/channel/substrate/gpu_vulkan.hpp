#pragma once

/**
 * @file gpu_vulkan.hpp
 * @brief Vulkan GPU substrate for zero-copy host-visible memory
 */

#include <psyne/config_detect.hpp>
#include <psyne/core/behaviors.hpp>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cstring>

#ifdef PSYNE_VULKAN_ENABLED
#include <vulkan/vulkan.h>
#endif

namespace psyne::substrate {

/**
 * @brief Vulkan GPU substrate implementation with host-visible memory
 * 
 * Provides zero-copy access to GPU memory through Vulkan's host-visible
 * and host-coherent memory types.
 */
class GPUVulkan : public behaviors::SubstrateBehavior {
public:
    /**
     * @brief Construct Vulkan substrate
     * @param device_index Physical device index to use
     */
    explicit GPUVulkan(int device_index = 0) : device_index_(device_index) {
#ifdef PSYNE_VULKAN_ENABLED
        init_vulkan();
#else
        throw std::runtime_error("Vulkan support not enabled. Rebuild with PSYNE_VULKAN_ENABLED");
#endif
    }
    
    ~GPUVulkan() {
        cleanup_allocations();
#ifdef PSYNE_VULKAN_ENABLED
        cleanup_vulkan();
#endif
    }
    
    void* allocate_memory_slab(size_t size_bytes) override {
#ifdef PSYNE_VULKAN_ENABLED
        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = size_bytes;
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT | 
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        VkBuffer buffer;
        if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan buffer");
        }
        
        // Get memory requirements
        VkMemoryRequirements mem_requirements;
        vkGetBufferMemoryRequirements(device_, buffer, &mem_requirements);
        
        // Find suitable memory type that is host visible and coherent
        uint32_t memory_type = find_memory_type(mem_requirements.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_requirements.size;
        alloc_info.memoryTypeIndex = memory_type;
        
        VkDeviceMemory device_memory;
        if (vkAllocateMemory(device_, &alloc_info, nullptr, &device_memory) != VK_SUCCESS) {
            vkDestroyBuffer(device_, buffer, nullptr);
            throw std::runtime_error("Failed to allocate Vulkan device memory");
        }
        
        // Bind buffer to memory
        vkBindBufferMemory(device_, buffer, device_memory, 0);
        
        // Map memory to get CPU pointer
        void* ptr = nullptr;
        if (vkMapMemory(device_, device_memory, 0, size_bytes, 0, &ptr) != VK_SUCCESS) {
            vkFreeMemory(device_, device_memory, nullptr);
            vkDestroyBuffer(device_, buffer, nullptr);
            throw std::runtime_error("Failed to map Vulkan memory");
        }
        
        allocations_.push_back({buffer, device_memory, ptr, size_bytes});
        return ptr;
#else
        throw std::runtime_error("Vulkan support not enabled");
#endif
    }
    
    void deallocate_memory_slab(void* memory) override {
#ifdef PSYNE_VULKAN_ENABLED
        if (!memory) return;
        
        auto it = std::find_if(allocations_.begin(), allocations_.end(),
                              [memory](const auto& alloc) { return alloc.mapped_ptr == memory; });
        
        if (it != allocations_.end()) {
            vkUnmapMemory(device_, it->device_memory);
            vkFreeMemory(device_, it->device_memory, nullptr);
            vkDestroyBuffer(device_, it->buffer, nullptr);
            allocations_.erase(it);
        }
#endif
    }
    
    void transport_send(void* data, size_t size) override {
        // GPU memory doesn't need network transport
        // This is for cross-GPU communication in the future
    }
    
    void transport_receive(void* buffer, size_t buffer_size) override {
        // GPU memory doesn't need network transport
        // This is for cross-GPU communication in the future
    }
    
    const char* substrate_name() const override { return "GPUVulkan"; }
    bool is_zero_copy() const override { return true; }
    bool is_cross_process() const override { return false; }
    
    /**
     * @brief Get the Vulkan device index
     */
    int get_device_index() const { return device_index_; }
    
    /**
     * @brief Flush CPU writes to ensure GPU visibility
     * @param ptr Memory pointer
     * @param size Memory size
     */
    void flush_memory(void* ptr, size_t size) {
#ifdef PSYNE_VULKAN_ENABLED
        // Host coherent memory doesn't need explicit flushing
        // This is a no-op but provided for API consistency
#endif
    }

private:
    int device_index_;
    
    struct Allocation {
        VkBuffer buffer;
        VkDeviceMemory device_memory;
        void* mapped_ptr;
        size_t size;
    };
    std::vector<Allocation> allocations_;
    
#ifdef PSYNE_VULKAN_ENABLED
    VkInstance instance_;
    VkPhysicalDevice physical_device_;
    VkDevice device_;
    VkPhysicalDeviceMemoryProperties memory_properties_;
    
    void init_vulkan() {
        // Create Vulkan instance
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "Psyne GPU Substrate";
        app_info.applicationVersion = VK_MAKE_VERSION(2, 0, 1);
        app_info.pEngineName = "Psyne";
        app_info.engineVersion = VK_MAKE_VERSION(2, 0, 1);
        app_info.apiVersion = VK_API_VERSION_1_2;
        
        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;
        
        if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan instance");
        }
        
        // Enumerate physical devices
        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
        
        if (device_count == 0) {
            throw std::runtime_error("No Vulkan devices found");
        }
        
        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
        
        if (device_index_ >= device_count) {
            throw std::runtime_error("Vulkan device index out of range");
        }
        
        physical_device_ = devices[device_index_];
        
        // Get memory properties
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties_);
        
        // Create logical device
        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_create_info{};
        queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info.queueFamilyIndex = 0; // Simplified - use first queue family
        queue_create_info.queueCount = 1;
        queue_create_info.pQueuePriorities = &queue_priority;
        
        VkPhysicalDeviceFeatures device_features{};
        
        VkDeviceCreateInfo device_create_info{};
        device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_create_info.pQueueCreateInfos = &queue_create_info;
        device_create_info.queueCreateInfoCount = 1;
        device_create_info.pEnabledFeatures = &device_features;
        
        if (vkCreateDevice(physical_device_, &device_create_info, nullptr, &device_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan logical device");
        }
    }
    
    void cleanup_vulkan() {
        if (device_) {
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_) {
            vkDestroyInstance(instance_, nullptr);
        }
    }
    
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
        for (uint32_t i = 0; i < memory_properties_.memoryTypeCount; i++) {
            if ((type_filter & (1 << i)) && 
                (memory_properties_.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("Failed to find suitable Vulkan memory type");
    }
#endif
    
    void cleanup_allocations() {
        for (auto& alloc : allocations_) {
            deallocate_memory_slab(alloc.mapped_ptr);
        }
        allocations_.clear();
    }
};

} // namespace psyne::substrate