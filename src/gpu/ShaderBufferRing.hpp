#pragma once

#include "ShaderBuffer.hpp"
#include "VulkanContext.hpp"
#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <cstdint>

namespace psyne {

/**
 * @brief Helper that manages a rotating set of ShaderBuffer input/output pairs.
 *
 * This is a small convenience class used to prepare for the buffering strategy
 * described in PIPELINE.md. Each pair contains one input and one output buffer
 * of fixed size. The ring enables uploads, dispatches and downloads to occur
 * concurrently so new input can be prepared while the GPU processes the
 * previous batch.
 */
class ShaderBufferRing {
  public:
    struct BufferPair {
        std::unique_ptr<ShaderBuffer> input;
        std::unique_ptr<ShaderBuffer> output;
    };

    ShaderBufferRing(uint32_t count, VkDeviceSize inSize, VkDeviceSize outSize,
                     uint32_t deviceIndex = 0)
        : deviceIndex_(deviceIndex) {
        auto& ctx = VulkanContext::getInstance(deviceIndex_);
        for (uint32_t i = 0; i < count; ++i) {
            ShaderBuffer::CreateInfo inCI{};
            inCI.name = "ring_input_" + std::to_string(i);
            inCI.device = ctx.getDevice();
            inCI.physicalDevice = ctx.getPhysicalDevice();
            inCI.size = inSize;
            inCI.memoryProperties =
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            inCI.access = ShaderBuffer::Access::InputOnly;
            auto inBuf = std::make_unique<ShaderBuffer>(inCI);

            ShaderBuffer::CreateInfo outCI{};
            outCI.name = "ring_output_" + std::to_string(i);
            outCI.device = ctx.getDevice();
            outCI.physicalDevice = ctx.getPhysicalDevice();
            outCI.size = outSize;
            outCI.memoryProperties =
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            outCI.access = ShaderBuffer::Access::OutputOnly;
            auto outBuf = std::make_unique<ShaderBuffer>(outCI);

            pairs_.push_back({std::move(inBuf), std::move(outBuf)});
        }
    }

    BufferPair& operator[](size_t idx) { return pairs_.at(idx); }
    const BufferPair& operator[](size_t idx) const { return pairs_.at(idx); }
    size_t size() const { return pairs_.size(); }

    /**
     * @brief Return the current buffer pair.
     */
    BufferPair& current() { return pairs_.at(current_); }

    /**
     * @brief Advance to the next pair and return it.
     */
    BufferPair& next() {
        current_ = (current_ + 1) % pairs_.size();
        return pairs_.at(current_);
    }

    /**
     * @brief Current index in the ring.
     */
    size_t index() const { return current_; }

  private:
    std::vector<BufferPair> pairs_;
    size_t current_ = 0;
    uint32_t deviceIndex_ = 0;
};

} // namespace harnomnics
