/**
 * @file vulkan_buffer.cpp
 * @brief Vulkan buffer implementation
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#include "vulkan_buffer.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace psyne {
namespace gpu {
namespace vulkan {

// VulkanBuffer implementation

VulkanBuffer::VulkanBuffer(VkDevice device, VkBuffer buffer,
                           VkDeviceMemory memory, size_t size,
                           BufferUsage usage, MemoryAccess access)
    : device_(device), buffer_(buffer), memory_(memory), size_(size),
      usage_(usage), access_(access), mapped_ptr_(nullptr) {}

VulkanBuffer::~VulkanBuffer() {
    if (mapped_ptr_) {
        unmap();
    }

    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, buffer_, nullptr);
    }

    if (memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, memory_, nullptr);
    }
}

void *VulkanBuffer::map() {
    if (mapped_ptr_) {
        return mapped_ptr_;
    }

    if (access_ == MemoryAccess::DeviceOnly) {
        throw std::runtime_error("Cannot map device-only buffer");
    }

    VkResult result = vkMapMemory(device_, memory_, 0, size_, 0, &mapped_ptr_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to map Vulkan buffer memory");
    }

    return mapped_ptr_;
}

void VulkanBuffer::unmap() {
    if (!mapped_ptr_) {
        return;
    }

    vkUnmapMemory(device_, memory_);
    mapped_ptr_ = nullptr;
}

void VulkanBuffer::flush() {
    if (!mapped_ptr_) {
        return;
    }

    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory_;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;

    vkFlushMappedMemoryRanges(device_, 1, &range);
}

void VulkanBuffer::upload(const void *data, size_t size, size_t offset) {
    if (offset + size > size_) {
        throw std::out_of_range("Upload exceeds buffer size");
    }

    void *dst = map();
    // Note: memcpy is necessary here for CPU->GPU memory transfer
    // This is a fundamental GPU operation and cannot be avoided
    std::memcpy(static_cast<uint8_t *>(dst) + offset, data, size);
    unmap();

    if (access_ != MemoryAccess::Shared) {
        flush();
    }
}

void VulkanBuffer::download(void *data, size_t size, size_t offset) {
    if (offset + size > size_) {
        throw std::out_of_range("Download exceeds buffer size");
    }

    void *src = map();
    // Note: memcpy is necessary here for GPU->CPU memory transfer
    // This is a fundamental GPU operation and cannot be avoided
    std::memcpy(data, static_cast<uint8_t *>(src) + offset, size);
    unmap();
}

// VulkanBufferFactory implementation

VulkanBufferFactory::VulkanBufferFactory(VkDevice device,
                                         VkPhysicalDevice physical_device,
                                         uint32_t queue_family_index)
    : device_(device), physical_device_(physical_device),
      queue_family_index_(queue_family_index) {
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties_);
}

VulkanBufferFactory::~VulkanBufferFactory() {}

std::unique_ptr<GPUBuffer>
VulkanBufferFactory::create_buffer(size_t size, BufferUsage usage,
                                   MemoryAccess access) {
    if (size == 0) {
        throw std::invalid_argument("Buffer size cannot be zero");
    }

    // Create buffer
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = get_buffer_usage_flags(usage, access);
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    buffer_info.queueFamilyIndexCount = 1;
    buffer_info.pQueueFamilyIndices = &queue_family_index_;

    VkBuffer buffer;
    if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan buffer");
    }

    // Get memory requirements
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device_, buffer, &mem_requirements);

    // Allocate memory
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = find_memory_type(
        mem_requirements.memoryTypeBits, get_memory_property_flags(access));

    VkDeviceMemory memory;
    if (vkAllocateMemory(device_, &alloc_info, nullptr, &memory) !=
        VK_SUCCESS) {
        vkDestroyBuffer(device_, buffer, nullptr);
        throw std::runtime_error("Failed to allocate Vulkan buffer memory");
    }

    // Bind memory to buffer
    if (vkBindBufferMemory(device_, buffer, memory, 0) != VK_SUCCESS) {
        vkFreeMemory(device_, memory, nullptr);
        vkDestroyBuffer(device_, buffer, nullptr);
        throw std::runtime_error("Failed to bind Vulkan buffer memory");
    }

    return std::make_unique<VulkanBuffer>(device_, buffer, memory, size, usage,
                                          access);
}

bool VulkanBufferFactory::supports_unified_memory() const {
    // Check for device coherent memory support
    for (uint32_t i = 0; i < memory_properties_.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags flags =
            memory_properties_.memoryTypes[i].propertyFlags;
        if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            return true;
        }
    }
    return false;
}

size_t VulkanBufferFactory::max_buffer_size() const {
    // Conservative estimate - actual limit depends on memory type
    return 1ULL << 30; // 1GB
}

uint32_t
VulkanBufferFactory::find_memory_type(uint32_t type_filter,
                                      VkMemoryPropertyFlags properties) {
    for (uint32_t i = 0; i < memory_properties_.memoryTypeCount; ++i) {
        if ((type_filter & (1 << i)) &&
            (memory_properties_.memoryTypes[i].propertyFlags & properties) ==
                properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type");
}

VkBufferUsageFlags
VulkanBufferFactory::get_buffer_usage_flags(BufferUsage usage,
                                            MemoryAccess access) {
    VkBufferUsageFlags flags =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    // Always enable compute usage for GPU buffers
    flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    // Enable uniform buffer for read-only access
    if (usage == BufferUsage::Static) {
        flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    }

    return flags;
}

VkMemoryPropertyFlags
VulkanBufferFactory::get_memory_property_flags(MemoryAccess access) {
    switch (access) {
    case MemoryAccess::DeviceOnly:
        return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    case MemoryAccess::HostOnly:
        return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    case MemoryAccess::Shared:
    case MemoryAccess::Managed:
        // Prefer device local memory that is also host visible
        return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }

    return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
}

// VulkanContext implementation

VulkanContext::VulkanContext()
    : instance_(VK_NULL_HANDLE), physical_device_(VK_NULL_HANDLE),
      device_(VK_NULL_HANDLE), compute_queue_(VK_NULL_HANDLE),
      command_pool_(VK_NULL_HANDLE), compute_queue_family_(0) {
    if (!initialize()) {
        throw std::runtime_error("Failed to initialize Vulkan context");
    }
}

VulkanContext::~VulkanContext() {
    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, command_pool_, nullptr);
    }

    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
    }

#ifdef DEBUG
    if (debug_messenger_ != VK_NULL_HANDLE) {
        // Would need to load extension function
    }
#endif

    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

bool VulkanContext::initialize() {
    if (!create_instance()) {
        return false;
    }

    if (!select_physical_device()) {
        return false;
    }

    if (!create_logical_device()) {
        return false;
    }

    // Create command pool
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = compute_queue_family_;

    if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) !=
        VK_SUCCESS) {
        return false;
    }

    return true;
}

bool VulkanContext::create_instance() {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Psyne";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 2, 2);
    app_info.pEngineName = "Psyne Engine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 2, 2);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    // Extensions
    std::vector<const char *> extensions;

#ifdef DEBUG
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    create_info.enabledExtensionCount =
        static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();

    // Validation layers
    std::vector<const char *> validation_layers;

#ifdef DEBUG
    validation_layers.push_back("VK_LAYER_KHRONOS_validation");
#endif

    create_info.enabledLayerCount =
        static_cast<uint32_t>(validation_layers.size());
    create_info.ppEnabledLayerNames = validation_layers.data();

    return vkCreateInstance(&create_info, nullptr, &instance_) == VK_SUCCESS;
}

bool VulkanContext::select_physical_device() {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);

    if (device_count == 0) {
        return false;
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

    // Select first suitable device
    for (const auto &device : devices) {
        vkGetPhysicalDeviceProperties(device, &device_properties_);
        vkGetPhysicalDeviceFeatures(device, &device_features_);
        vkGetPhysicalDeviceMemoryProperties(device, &memory_properties_);

        // Check for compute queue
        if (find_compute_queue_family() != UINT32_MAX) {
            physical_device_ = device;
            return true;
        }
    }

    return false;
}

bool VulkanContext::create_logical_device() {
    compute_queue_family_ = find_compute_queue_family();
    if (compute_queue_family_ == UINT32_MAX) {
        return false;
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = compute_queue_family_;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceFeatures device_features{};
    // Enable features as needed

    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_create_info;
    create_info.pEnabledFeatures = &device_features;

    if (vkCreateDevice(physical_device_, &create_info, nullptr, &device_) !=
        VK_SUCCESS) {
        return false;
    }

    vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);
    return true;
}

uint32_t VulkanContext::find_compute_queue_family() {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_,
                                             &queue_family_count, nullptr);

    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device_, &queue_family_count, queue_families.data());

    for (uint32_t i = 0; i < queue_family_count; ++i) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            return i;
        }
    }

    return UINT32_MAX;
}

std::unique_ptr<GPUBufferFactory> VulkanContext::create_buffer_factory() {
    return std::make_unique<VulkanBufferFactory>(device_, physical_device_,
                                                 compute_queue_family_);
}

std::string VulkanContext::device_name() const {
    return device_properties_.deviceName;
}

size_t VulkanContext::total_memory() const {
    size_t total = 0;
    for (uint32_t i = 0; i < memory_properties_.memoryHeapCount; ++i) {
        if (memory_properties_.memoryHeaps[i].flags &
            VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            total += memory_properties_.memoryHeaps[i].size;
        }
    }
    return total;
}

size_t VulkanContext::available_memory() const {
    // Vulkan doesn't provide available memory info directly
    // Would need to track allocations or use extensions
    return total_memory(); // Approximation
}

bool VulkanContext::is_unified_memory() const {
    // Check for integrated GPU
    return device_properties_.deviceType ==
           VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
}

void VulkanContext::synchronize() {
    vkQueueWaitIdle(compute_queue_);
}

VkShaderModule
VulkanContext::load_shader_module(const std::vector<uint32_t> &spirv_code) {
    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = spirv_code.size() * sizeof(uint32_t);
    create_info.pCode = spirv_code.data();

    VkShaderModule shader_module;
    if (vkCreateShaderModule(device_, &create_info, nullptr, &shader_module) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }

    return shader_module;
}

VkPipeline
VulkanContext::create_compute_pipeline(VkShaderModule shader_module,
                                       VkPipelineLayout pipeline_layout) {
    VkPipelineShaderStageCreateInfo shader_stage_info{};
    shader_stage_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shader_stage_info.module = shader_module;
    shader_stage_info.pName = "main";

    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = shader_stage_info;
    pipeline_info.layout = pipeline_layout;

    VkPipeline pipeline;
    if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info,
                                 nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline");
    }

    return pipeline;
}

// VulkanCommandBuffer implementation

VulkanCommandBuffer::VulkanCommandBuffer(VkDevice device, VkCommandPool pool)
    : device_(device), pool_(pool) {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device_, &alloc_info, &buffer_) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffer");
    }
}

VulkanCommandBuffer::~VulkanCommandBuffer() {
    if (buffer_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(device_, pool_, 1, &buffer_);
    }
}

void VulkanCommandBuffer::begin() {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(buffer_, &begin_info) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin command buffer");
    }
}

void VulkanCommandBuffer::end() {
    if (vkEndCommandBuffer(buffer_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end command buffer");
    }
}

void VulkanCommandBuffer::submit(VkQueue queue, VkFence fence) {
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &buffer_;

    if (vkQueueSubmit(queue, 1, &submit_info, fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer");
    }
}

void VulkanCommandBuffer::bind_pipeline(VkPipeline pipeline) {
    vkCmdBindPipeline(buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
}

void VulkanCommandBuffer::bind_descriptor_set(VkPipelineLayout layout,
                                              VkDescriptorSet set) {
    vkCmdBindDescriptorSets(buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0,
                            1, &set, 0, nullptr);
}

void VulkanCommandBuffer::dispatch(uint32_t x, uint32_t y, uint32_t z) {
    vkCmdDispatch(buffer_, x, y, z);
}

void VulkanCommandBuffer::copy_buffer(VkBuffer src, VkBuffer dst,
                                      VkDeviceSize size) {
    VkBufferCopy copy_region{};
    copy_region.size = size;
    vkCmdCopyBuffer(buffer_, src, dst, 1, &copy_region);
}

void VulkanCommandBuffer::fill_buffer(VkBuffer buffer, VkDeviceSize offset,
                                      VkDeviceSize size, uint32_t data) {
    vkCmdFillBuffer(buffer_, buffer, offset, size, data);
}

void VulkanCommandBuffer::pipeline_barrier(VkPipelineStageFlags src_stage,
                                           VkPipelineStageFlags dst_stage,
                                           VkBufferMemoryBarrier *barrier) {
    vkCmdPipelineBarrier(buffer_, src_stage, dst_stage, 0, 0, nullptr, 1,
                         barrier, 0, nullptr);
}

} // namespace vulkan
} // namespace gpu
} // namespace psyne