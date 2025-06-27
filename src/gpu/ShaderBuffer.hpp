#pragma once

#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>
#include "GPUBuffer.hpp"
#include "VulkanContext.hpp"

namespace psyne {

/**
 * @brief Finds a suitable memory type for a buffer.
 *
 * @param physDev The physical device to query.
 * @param typeFilter The memory type filter.
 * @param props The memory properties to match.
 * @return uint32_t The memory type index.
 */
static uint32_t findMemoryType(VkPhysicalDevice physDev, uint32_t typeFilter,
                               VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && ((memProps.memoryTypes[i].propertyFlags & props) == props)) {
            return i;
        }
    }
    throw std::runtime_error("ShaderBuffer: No suitable memory type found.");
}

/**
 * @class ShaderBuffer
 *
 * @brief Represents a GPU-visible Vulkan buffer used in shader stages.
 * Can be linked to a CPU-side GPUBuffer or operated independently.
 */
class ShaderBuffer {
public:
    enum class Access { InputOnly, OutputOnly, InputOutput };

    struct CreateInfo {
        std::string name;
        VkDevice device;
        VkPhysicalDevice physicalDevice;
        VkDeviceSize size;
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        Access access = Access::InputOutput;
        VkQueue queue = VK_NULL_HANDLE;
        uint32_t queueFamilyIndex = 0;
    };

    /**
     * @brief Construct a new ShaderBuffer object
     */
    ShaderBuffer(const CreateInfo& ci)
        : device_(ci.device), physicalDevice_(ci.physicalDevice), buffer_(VK_NULL_HANDLE),
          memory_(VK_NULL_HANDLE), size_(ci.size), name_(ci.name), access_(ci.access),
          usage_(ci.usage), memoryProperties_(ci.memoryProperties),
          queueFamilyIndex_(ci.queueFamilyIndex), queue_(ci.queue) {
        allocateBuffer();
        allocateMemory();
        bindMemory();
        setDebugName(ci.name);
    }

    /**
     * @brief Destroy the ShaderBuffer object
     */
    ~ShaderBuffer() {
        if (buffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, buffer_, nullptr);
            buffer_ = VK_NULL_HANDLE;
        }
        if (memory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, memory_, nullptr);
            memory_ = VK_NULL_HANDLE;
        }
    }

    ShaderBuffer(const ShaderBuffer&) = delete;
    ShaderBuffer& operator=(const ShaderBuffer&) = delete;

    /**
     * @brief Move constructor
     */
    ShaderBuffer(ShaderBuffer&& other) noexcept
        : device_(other.device_), physicalDevice_(other.physicalDevice_), buffer_(other.buffer_),
          memory_(other.memory_), size_(other.size_), name_(std::move(other.name_)),
          access_(other.access_), usage_(other.usage_), memoryProperties_(other.memoryProperties_),
          queueFamilyIndex_(other.queueFamilyIndex_), queue_(other.queue_),
          linkedInput_(std::move(other.linkedInput_)),
          linkedOutput_(std::move(other.linkedOutput_)) {
        other.buffer_ = VK_NULL_HANDLE;
        other.memory_ = VK_NULL_HANDLE;
        other.size_ = 0;
    }

    /**
     * @brief Move assignment operator
     */
    ShaderBuffer& operator=(ShaderBuffer&& other) noexcept {
        if (this != &other) {
            // Cleanup our current resources
            if (buffer_ != VK_NULL_HANDLE)
                vkDestroyBuffer(device_, buffer_, nullptr);
            if (memory_ != VK_NULL_HANDLE)
                vkFreeMemory(device_, memory_, nullptr);

            // Move over
            device_ = other.device_;
            physicalDevice_ = other.physicalDevice_;
            buffer_ = other.buffer_;
            memory_ = other.memory_;
            size_ = other.size_;
            name_ = std::move(other.name_);
            access_ = other.access_;
            usage_ = other.usage_;
            memoryProperties_ = other.memoryProperties_;
            queueFamilyIndex_ = other.queueFamilyIndex_;
            queue_ = other.queue_;
            linkedInput_ = std::move(other.linkedInput_);
            linkedOutput_ = std::move(other.linkedOutput_);

            other.buffer_ = VK_NULL_HANDLE;
            other.memory_ = VK_NULL_HANDLE;
            other.size_ = 0;
        }
        return *this;
    }

    // Core buffer info
    VkBuffer getBuffer() const { return buffer_; }
    VkDeviceMemory getMemory() const { return memory_; }
    VkDeviceSize getSize() const { return size_; }
    VkDevice getDevice() const { return device_; }
    const std::string& getName() const { return name_; }

    /**
     * @brief Upload data from the host to the GPU buffer.
     */
    void uploadFromHost(const void* data, size_t size, size_t offset = 0) {
        if (!data || size == 0)
            return;
        if (offset + size > size_) {
            throw std::runtime_error("ShaderBuffer::uploadFromHost: Out of range.");
        }

        // If memory is host-visible, map directly
        bool hostVisible = (memoryProperties_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        if (hostVisible) {
            void* mapped = nullptr;
            vkMapMemory(device_, memory_, offset, size, 0, &mapped);
            std::memcpy(mapped, data, size);
            vkUnmapMemory(device_, memory_);
        } else {
            // Production approach: create a staging buffer, map & copy, then do GPU copy
            // This is a basic version:
            VkBuffer stagingBuf;
            VkDeviceMemory stagingMem;
            {
                // create staging
                VkBufferCreateInfo bci{};
                bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bci.size = size;
                bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                if (vkCreateBuffer(device_, &bci, nullptr, &stagingBuf) != VK_SUCCESS) {
                    throw std::runtime_error("ShaderBuffer: create staging buffer failed.");
                }
                VkMemoryRequirements req;
                vkGetBufferMemoryRequirements(device_, stagingBuf, &req);
                VkMemoryAllocateInfo ai{};
                ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                ai.allocationSize = req.size;
                ai.memoryTypeIndex = findMemoryType(physicalDevice_, req.memoryTypeBits,
                                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                if (vkAllocateMemory(device_, &ai, nullptr, &stagingMem) != VK_SUCCESS) {
                    vkDestroyBuffer(device_, stagingBuf, nullptr);
                    throw std::runtime_error("ShaderBuffer: staging alloc failed.");
                }
                vkBindBufferMemory(device_, stagingBuf, stagingMem, 0);
                // map & copy
                void* mapped = nullptr;
                vkMapMemory(device_, stagingMem, 0, size, 0, &mapped);
                std::memcpy(mapped, data, size);
                vkUnmapMemory(device_, stagingMem);
            }
            // copy from staging to this->buffer_
            VkCommandPoolCreateInfo cpi{};
            cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            cpi.queueFamilyIndex = queueFamilyIndex_;
            cpi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

            VkCommandPool cmdPool;
            vkCreateCommandPool(device_, &cpi, nullptr, &cmdPool);

            VkCommandBufferAllocateInfo cai{};
            cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cai.commandPool = cmdPool;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 1;

            VkCommandBuffer cmdBuf;
            vkAllocateCommandBuffers(device_, &cai, &cmdBuf);

            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmdBuf, &bi);

            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = offset;
            copyRegion.size = size;
            vkCmdCopyBuffer(cmdBuf, stagingBuf, buffer_, 1, &copyRegion);

            vkEndCommandBuffer(cmdBuf);

            VkFenceCreateInfo fci{};
            fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            VkFence fence;
            vkCreateFence(device_, &fci, nullptr, &fence);

            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmdBuf;

            vkQueueSubmit(queue_, 1, &si, fence);
            vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);

            vkDestroyFence(device_, fence, nullptr);
            vkFreeCommandBuffers(device_, cmdPool, 1, &cmdBuf);
            vkDestroyCommandPool(device_, cmdPool, nullptr);

            vkDestroyBuffer(device_, stagingBuf, nullptr);
            vkFreeMemory(device_, stagingMem, nullptr);
        }
    }

    /**
     * @brief Download data from the GPU buffer to the host.
     */
    void downloadToHost(void* dst, size_t size, size_t offset = 0) {
        if (!dst || size == 0)
            return;
        if (offset + size > size_) {
            throw std::runtime_error("ShaderBuffer::downloadToHost: Out of range.");
        }
        bool hostVisible = (memoryProperties_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        if (hostVisible) {
            // direct map
            void* mapped = nullptr;
            vkMapMemory(device_, memory_, offset, size, 0, &mapped);
            std::memcpy(dst, mapped, size);
            vkUnmapMemory(device_, memory_);
        } else {
            // create staging buffer, copy from device, map
            VkBuffer stagingBuf;
            VkDeviceMemory stagingMem;
            {
                VkBufferCreateInfo bci{};
                bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bci.size = size;
                bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                if (vkCreateBuffer(device_, &bci, nullptr, &stagingBuf) != VK_SUCCESS) {
                    throw std::runtime_error(
                        "ShaderBuffer: create staging buffer for download failed.");
                }
                VkMemoryRequirements req;
                vkGetBufferMemoryRequirements(device_, stagingBuf, &req);
                VkMemoryAllocateInfo ai{};
                ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                ai.allocationSize = req.size;
                ai.memoryTypeIndex = findMemoryType(physicalDevice_, req.memoryTypeBits,
                                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                if (vkAllocateMemory(device_, &ai, nullptr, &stagingMem) != VK_SUCCESS) {
                    vkDestroyBuffer(device_, stagingBuf, nullptr);
                    throw std::runtime_error("ShaderBuffer: staging alloc for download failed.");
                }
                vkBindBufferMemory(device_, stagingBuf, stagingMem, 0);
            }
            // copy from this->buffer_ to staging
            VkCommandPoolCreateInfo cpi{};
            cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            cpi.queueFamilyIndex = queueFamilyIndex_;
            cpi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            VkCommandPool cmdPool;
            vkCreateCommandPool(device_, &cpi, nullptr, &cmdPool);

            VkCommandBufferAllocateInfo cai{};
            cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cai.commandPool = cmdPool;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 1;
            VkCommandBuffer cmdBuf;
            vkAllocateCommandBuffers(device_, &cai, &cmdBuf);

            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmdBuf, &bi);

            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = offset;
            copyRegion.dstOffset = 0;
            copyRegion.size = size;
            vkCmdCopyBuffer(cmdBuf, buffer_, stagingBuf, 1, &copyRegion);

            vkEndCommandBuffer(cmdBuf);

            VkFenceCreateInfo fci{};
            fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            VkFence fence;
            vkCreateFence(device_, &fci, nullptr, &fence);

            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmdBuf;

            vkQueueSubmit(queue_, 1, &si, fence);
            vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);

            vkDestroyFence(device_, fence, nullptr);
            vkFreeCommandBuffers(device_, cmdPool, 1, &cmdBuf);
            vkDestroyCommandPool(device_, cmdPool, nullptr);

            // map staging
            void* mapped = nullptr;
            vkMapMemory(device_, stagingMem, 0, size, 0, &mapped);
            std::memcpy(dst, mapped, size);
            vkUnmapMemory(device_, stagingMem);

            vkDestroyBuffer(device_, stagingBuf, nullptr);
            vkFreeMemory(device_, stagingMem, nullptr);
        }
    }

    /**
     * @brief Bind an input buffer to this buffer.
     */
    void bindInputFrom(GPUBuffer& buffer) { linkedInput_ = &buffer; }

    /**
     * @brief Bind an output buffer to this buffer.
     */
    void bindOutputTo(GPUBuffer& buffer) { linkedOutput_ = &buffer; }

    /**
     * @brief Flush data from the linked input buffer to the GPU buffer.
     */
    void flushToGPU() {
        // Production code might read from the linkedInput_ buffer in chunks,
        // then upload to the GPU. We'll do a simple approach:
        if (!linkedInput_ || access_ == Access::OutputOnly)
            return;
        // Example read:
        auto chunk = linkedInput_->get();
        auto chunkPtr = chunk.get();
        if (chunkPtr && !chunkPtr->empty()) {
            // parse (*chunkPtr)[0] -> if float, etc.
            // For brevity, assume a single float
            if constexpr (std::is_same_v<std::decay_t<decltype((*chunkPtr)[0])>, float>) {
                float val = (*chunkPtr)[0];
                uploadFromHost(&val, sizeof(val));
            }
        }
    }

    /**
     * @brief Fetch data from the GPU buffer to the linked output buffer.
     */
    void fetchFromGPU() {
        if (!linkedOutput_ || access_ == Access::InputOnly)
            return;
        // Example single float
        float val = 0.f;
        downloadToHost(&val, sizeof(val));
        linkedOutput_.value()->push(std::make_shared<std::vector<float>>(val));
    }

    /**
     * @brief Bind the buffer as a descriptor.
     */
    void bindAsDescriptor(uint32_t binding, VkDescriptorSet descSet) const {
        VkDescriptorBufferInfo dbi{};
        dbi.buffer = buffer_;
        dbi.offset = 0;
        dbi.range = size_;

        VkWriteDescriptorSet wds{};
        wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds.dstSet = descSet;
        wds.dstBinding = binding;
        wds.descriptorCount = 1;
        wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds.pBufferInfo = &dbi;

        vkUpdateDescriptorSets(device_, 1, &wds, 0, nullptr);
    }

    /**
     * @brief Record a barrier for the buffer.
     */
    void recordBarrier(VkCommandBuffer cmdBuf, VkPipelineStageFlags srcStage,
                       VkPipelineStageFlags dstStage) const {
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        barrier.srcQueueFamilyIndex = queueFamilyIndex_;
        barrier.dstQueueFamilyIndex = queueFamilyIndex_;
        barrier.buffer = buffer_;
        barrier.offset = 0;
        barrier.size = size_;

        vkCmdPipelineBarrier(cmdBuf, srcStage, dstStage, 0, 0, nullptr, 1, &barrier, 0, nullptr);
    }

    /**
     * @brief Set a debug name for the buffer.
     */
    void setDebugName(const std::string& name) {
        // If VK_EXT_debug_utils is enabled, we can name the buffer/memory
        // For brevity, check if function is available, etc.
        name_ = name;
#ifdef VK_EXT_debug_utils
        if (!name.empty()) {
            auto func = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
                vkGetDeviceProcAddr(device_, "vkSetDebugUtilsObjectNameEXT"));
            if (func) {
                VkDebugUtilsObjectNameInfoEXT nameInfo{};
                nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
                nameInfo.objectHandle = reinterpret_cast<uint64_t>(buffer_);
                nameInfo.objectType = VK_OBJECT_TYPE_BUFFER;
                nameInfo.pObjectName = name_.c_str();
                func(device_, &nameInfo);
            }
        }
#endif
    }

    /**
     * @brief Check if the buffer is mapped.
     */
    bool isMapped() const {
        // Not tracking persistent maps here. Real code might track them.
        return false;
    }

    /**
     * @brief Check if the buffer is host-visible.
     */
    bool isDeviceLocal() const {
        return (memoryProperties_ & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    }

    // --- simple accessors for slab helpers -----------------
    void* map()    { return persistentMap(); }      // see below
    VkBuffer handle() const { return buffer_; }

  private:
    void* persistentMap() {
        // Lazily map once, keep pointer for life of the buffer.
        if (!mapped_) {
            if (!(memoryProperties_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
                throw std::runtime_error("ShaderBuffer: buffer not host visible");
            vkMapMemory(device_, memory_, 0, size_, 0, &mapped_);
        }
        return mapped_;
    }

    void* mapped_ = nullptr;

    /**
     * @brief Allocate a buffer on the GPU.
     */
    void allocateBuffer() {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size_;
        bci.usage = usage_;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bci.queueFamilyIndexCount = 1;
        bci.pQueueFamilyIndices = &queueFamilyIndex_;

        if (vkCreateBuffer(device_, &bci, nullptr, &buffer_) != VK_SUCCESS) {
            throw std::runtime_error("ShaderBuffer: Failed to create buffer.");
        }
    }

    /**
     * @brief Allocate memory for the buffer.
     */
    void allocateMemory() {
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device_, buffer_, &req);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemoryType(physicalDevice_, req.memoryTypeBits, memoryProperties_);

        if (vkAllocateMemory(device_, &ai, nullptr, &memory_) != VK_SUCCESS) {
            throw std::runtime_error("ShaderBuffer: Failed to allocate buffer memory.");
        }
    }

    /**
     * @brief Bind the buffer to the allocated memory.
     */
    void bindMemory() {
        if (vkBindBufferMemory(device_, buffer_, memory_, 0) != VK_SUCCESS) {
            throw std::runtime_error("ShaderBuffer: Failed to bind buffer memory.");
        }
    }

    VkDevice device_;
    VkPhysicalDevice physicalDevice_;
    VkBuffer buffer_;
    VkDeviceMemory memory_;
    VkDeviceSize size_;
    std::string name_;
    Access access_;

    VkBufferUsageFlags usage_;
    VkMemoryPropertyFlags memoryProperties_;
    uint32_t queueFamilyIndex_;
    VkQueue queue_;
    GPUBuffer* linkedInput_ = nullptr;
    std::optional<GPUBuffer*> linkedOutput_;
};

} // namespace psyne