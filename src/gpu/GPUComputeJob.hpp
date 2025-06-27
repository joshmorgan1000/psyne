#pragma once

#include "GPUBuffer.hpp"
#include "GPUFunction.hpp"
#include "PipelineManager.hpp"
#include "ShaderBuffer.hpp"
#include "ShaderBufferRing.hpp"
#include "VulkanContext.hpp"
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <thread>
#include <utility>
#include <zstd.h>
#include <variant>
#include <vector>
#include <cstdint>

namespace psyne {

/**
 * @brief Utility to get the size in bytes for a std::vector<float>.
 *        Real production code should handle all variant types carefully,
 *        including nested vectors. Here we handle a few common cases.
 */
static size_t getVariantSizeInBytes(const std::vector<float>& var) {
    if (std::holds_alternative<std::vector<float>>(var)) {
        return std::get<std::vector<float>>(var).size() * sizeof(float);
    } else if (std::holds_alternative<std::vector<double>>(var)) {
        return std::get<std::vector<double>>(var).size() * sizeof(double);
    } else if (std::holds_alternative<std::vector<std::vector<float>>>(var)) {
        const auto& vec = std::get<std::vector<std::vector<float>>>(var);
        size_t bytes = 0;
        for (const auto& v : vec) {
            bytes += v.size() * sizeof(float);
        }
        return bytes;
    } else if (std::holds_alternative<std::vector<std::vector<double>>>(var)) {
        const auto& vec = std::get<std::vector<std::vector<double>>>(var);
        size_t bytes = 0;
        for (const auto& v : vec) {
            bytes += v.size() * sizeof(double);
        }
        return bytes;
    } 
    throw std::runtime_error("getVariantSizeInBytes: Unsupported variant type");
}

/**
 * @brief Utility to copy the contents of a std::vector<float> into a mapped memory region.
 */
static void copyVariantToMemory(const std::vector<float>& var, uint8_t* dst) {
    if (!dst)
        return;
    if (std::holds_alternative<float>(var)) {
        float v = std::get<float>(var);
        std::memcpy(dst, &v, sizeof(float));
    } else if (std::holds_alternative<double>(var)) {
        double v = std::get<double>(var);
        std::memcpy(dst, &v, sizeof(double));
    } else if (std::holds_alternative<std::uint16_t>(var)) {
        std::uint16_t v = std::get<std::uint16_t>(var);
        std::memcpy(dst, &v, sizeof(std::uint16_t));
    } else if (std::holds_alternative<std::int16_t>(var)) {
        std::int16_t v = std::get<std::int16_t>(var);
        std::memcpy(dst, &v, sizeof(std::int16_t));
    } else if (std::holds_alternative<std::int32_t>(var)) {
        std::int32_t v = std::get<std::int32_t>(var);
        std::memcpy(dst, &v, sizeof(std::int32_t));
    } else if (std::holds_alternative<std::uint32_t>(var)) {
        std::uint32_t v = std::get<std::uint32_t>(var);
        std::memcpy(dst, &v, sizeof(std::uint32_t));
    } else if (std::holds_alternative<std::uint64_t>(var)) {
        std::uint64_t v = std::get<std::uint64_t>(var);
        std::memcpy(dst, &v, sizeof(std::uint64_t));
    } else if (std::holds_alternative<std::int64_t>(var)) {
        std::int64_t v = std::get<std::int64_t>(var);
        std::memcpy(dst, &v, sizeof(std::int64_t));
    } else if (std::holds_alternative<std::vector<float>>(var)) {
        const auto& vec = std::get<std::vector<float>>(var);
        std::memcpy(dst, vec.data(), vec.size() * sizeof(float));
    } else if (std::holds_alternative<std::vector<double>>(var)) {
        const auto& vec = std::get<std::vector<double>>(var);
        std::memcpy(dst, vec.data(), vec.size() * sizeof(double));
    } else if (std::holds_alternative<std::vector<uint16_t>>(var)) {
        const auto& vec = std::get<std::vector<std::uint16_t>>(var);
        std::memcpy(dst, vec.data(), vec.size() * sizeof(uint16_t));
    } else if (std::holds_alternative<std::vector<std::int16_t>>(var)) {
        const auto& vec = std::get<std::vector<std::int16_t>>(var);
        std::memcpy(dst, vec.data(), vec.size() * sizeof(std::int16_t));
    } else if (std::holds_alternative<std::vector<std::int32_t>>(var)) {
        const auto& vec = std::get<std::vector<std::int32_t>>(var);
        std::memcpy(dst, vec.data(), vec.size() * sizeof(std::int32_t));
    } else if (std::holds_alternative<std::vector<std::uint32_t>>(var)) {
        const auto& vec = std::get<std::vector<std::uint32_t>>(var);
        std::memcpy(dst, vec.data(), vec.size() * sizeof(uint32_t));
    } else if (std::holds_alternative<std::vector<std::uint64_t>>(var)) {
        const auto& vec = std::get<std::vector<std::uint64_t>>(var);
        std::memcpy(dst, vec.data(), vec.size() * sizeof(std::uint64_t));
    } else if (std::holds_alternative<std::vector<std::int64_t>>(var)) {
        const auto& vec = std::get<std::vector<std::int64_t>>(var);
        std::memcpy(dst, vec.data(), vec.size() * sizeof(std::int64_t));
    } else if (std::holds_alternative<std::vector<std::vector<float>>>(var)) {
        const auto& vec = std::get<std::vector<std::vector<float>>>(var);
        size_t bytes = 0;
        for (const auto& v : vec) {
            size_t vbytes = v.size() * sizeof(float);
            std::memcpy(dst + bytes, v.data(), vbytes);
            bytes += vbytes;
        }
    } else if (std::holds_alternative<std::vector<std::vector<double>>>(var)) {
        const auto& vec = std::get<std::vector<std::vector<double>>>(var);
        size_t bytes = 0;
        for (const auto& v : vec) {
            size_t vbytes = v.size() * sizeof(double);
            std::memcpy(dst + bytes, v.data(), vbytes);
            bytes += vbytes;
        }
    } else if (std::holds_alternative<std::vector<std::vector<uint16_t>>>(var)) {
        const auto& vec = std::get<std::vector<std::vector<uint16_t>>>(var);
        size_t bytes = 0;
        for (const auto& v : vec) {
            size_t vbytes = v.size() * sizeof(uint16_t);
            std::memcpy(dst + bytes, v.data(), vbytes);
            bytes += vbytes;
        }
    } else if (std::holds_alternative<std::vector<std::vector<int16_t>>>(var)) {
        const auto& vec = std::get<std::vector<std::vector<int16_t>>>(var);
        size_t bytes = 0;
        for (const auto& v : vec) {
            size_t vbytes = v.size() * sizeof(int16_t);
            std::memcpy(dst + bytes, v.data(), vbytes);
            bytes += vbytes;
        }
    } else if (std::holds_alternative<std::vector<std::vector<int32_t>>>(var)) {
        const auto& vec = std::get<std::vector<std::vector<int32_t>>>(var);
        size_t bytes = 0;
        for (const auto& v : vec) {
            size_t vbytes = v.size() * sizeof(int32_t);
            std::memcpy(dst + bytes, v.data(), vbytes);
            bytes += vbytes;
        }
    } else if (std::holds_alternative<std::vector<std::vector<uint32_t>>>(var)) {
        const auto& vec = std::get<std::vector<std::vector<uint32_t>>>(var);
        size_t bytes = 0;
        for (const auto& v : vec) {
            size_t vbytes = v.size() * sizeof(uint32_t);
            std::memcpy(dst + bytes, v.data(), vbytes);
            bytes += vbytes;
        }
    } else if (std::holds_alternative<std::vector<std::vector<uint64_t>>>(var)) {
        const auto& vec = std::get<std::vector<std::vector<uint64_t>>>(var);
        size_t bytes = 0;
        for (const auto& v : vec) {
            size_t vbytes = v.size() * sizeof(uint64_t);
            std::memcpy(dst + bytes, v.data(), vbytes);
            bytes += vbytes;
        }
    } else if (std::holds_alternative<std::vector<std::vector<int64_t>>>(var)) {
        const auto& vec = std::get<std::vector<std::vector<int64_t>>>(var);
        size_t bytes = 0;
        for (const auto& v : vec) {
            size_t vbytes = v.size() * sizeof(int64_t);
            std::memcpy(dst + bytes, v.data(), vbytes);
            bytes += vbytes;
        }
    } else {
        throw std::runtime_error("copyVariantToMemory: Unsupported variant type");
    }
}

class GPUComputeJob {
  public:
    GPUComputeJob(const std::string& kernelName, std::shared_ptr<GPUBuffer> outputBuffer,
                  std::vector<std::shared_ptr<GPUBuffer>> inputBuffers,
                  const std::vector<uint32_t>& push_constants, float mem_percent,
                  uint32_t ringSize = 1, uint32_t deviceIndex = 0)
        : kernelName_(kernelName), output_buffer_(std::move(outputBuffer)),
          input_buffers_(std::move(inputBuffers)), push_constants_(push_constants),
          mem_percent_(mem_percent), ring_size_(ringSize), device_index_(deviceIndex) {}

    ~GPUComputeJob() { wait(); }

    // Non-copyable
    GPUComputeJob(const GPUComputeJob&) = delete;
    GPUComputeJob& operator=(const GPUComputeJob&) = delete;

    /**
     * @brief Start the compute job.
     */
    void start() {
        if (started_)
            return;
        started_ = true;
        worker_ = std::thread(&GPUComputeJob::threadMain, this);
    }

    /**
     * @brief Wait for the compute job to finish.
     */
    void wait() {
        if (started_ && worker_.joinable()) {
            worker_.join();
        }
    }

    /**
     * @brief Check if the compute job is done.
     */
    bool isDone() { return done_.load(); }

    /**
     * @brief Number of rotating buffer pairs used by this job.
     */
    uint32_t getRingSize() const { return ring_size_; }

  private:
    /**
     * @brief The main thread function for the compute job.
     */
    void threadMain() {
        auto cleanupWrapper = [this]() { cleanupVulkanResources(); };
        try {
            auto& ctx = VulkanContext::getInstance(device_index_);
            const auto* func = GPUFunctionRegistry::getInstance().get(kernelName_);
            gx_ = func->groupCountX_;
            gy_ = func->groupCountY_;
            gz_ = func->groupCountZ_;
            if (!func) {
                output_buffer_->push(nullptr);
                done_ = true;
                return;
            }

            size_t paramCount = func->parameters.size();
            // CPU fallback if GPU not viable or missing SPIR-V
            if (!ctx.isGPUCapable() || func->shader.empty()) {
                runCPUFallback(*func, paramCount);
                done_ = true;
                return;
            }

            // Get the target buffer size(ish) based on free memory
            size_t buffer_size = getFreeMem(ctx, mem_percent_);

            // Get one record so we know the record size
            std::vector<uint8_t> stagingBuffer;
            size_t record_size = 0;
            size_t offset = 0;
            for (size_t i = 0; i < paramCount; i++) {
                std::vector<std::shared_ptr<std::vector<float>>> item = input_buffers_[i]->getForGPU(1);
                for (size_t j = 0; j < item.size(); j++) {
                    size_t one_record_size = getVariantSizeInBytes(*item[j]);
                    std::vector<uint8_t> tmp(one_record_size);
                    copyVariantToMemory(*item[j], tmp.data());
                    stagingBuffer.resize(stagingBuffer.size() + one_record_size);
                    std::memcpy(stagingBuffer.data() + offset, tmp.data(), one_record_size);
                    offset += one_record_size;
                    record_size += one_record_size;
                }
            }

            // Fill the staging buffer with as many records as possible
            size_t iterations = 1;
            size_t current_record_size = record_size;
            bool endOfStream = false;
            while (stagingBuffer.size() + record_size < buffer_size) {
                if (endOfStream)
                    break;
                current_record_size = 0;
                for (size_t i = 0; i < paramCount; i++) {
                    if (endOfStream)
                        break;
                    std::vector<std::shared_ptr<std::vector<float>>> item =
                        input_buffers_[i]->getForGPU(1);
                    iterations++;
                    for (size_t j = 0; j < item.size(); j++) {
                        if (!item[j]) {
                            endOfStream = true;
                            break;
                        }
                        size_t one_record_size = getVariantSizeInBytes(*item[j]);
                        std::vector<uint8_t> tmp(one_record_size);
                        copyVariantToMemory(*item[j], tmp.data());
                        stagingBuffer.resize(stagingBuffer.size() + one_record_size);
                        std::memcpy(stagingBuffer.data() + offset, tmp.data(), one_record_size);
                        offset += one_record_size;
                        current_record_size += one_record_size;
                    }
                }
            }
            endOfStream = false;
            record_size = current_record_size;

            std::vector<uint8_t> compressed = func->shader;
            std::vector<uint8_t> decompressed;
            size_t decompressed_size =
                ZSTD_getFrameContentSize(compressed.data(), compressed.size());
            decompressed.resize(decompressed_size);
            size_t decompressed_actual = ZSTD_decompress(decompressed.data(), decompressed_size,
                                                         compressed.data(), compressed.size());
            if (ZSTD_isError(decompressed_actual)) {
                throw std::runtime_error("GPUComputeJob: decompression failed.");
            }

            // Create/find pipeline
            auto pipeline = ctx.pipelineManager().getOrCreate(kernelName_, decompressed, {},
                                                              func->push_constants_);
            if (!pipeline) {
                runCPUFallback(*func, paramCount);
                done_ = true;
                return;
            }

            // allocate rotating buffers
            buffer_ring_ = std::make_unique<ShaderBufferRing>(
                ring_size_, stagingBuffer.size(), iterations * func->output_variant_size_bytes_,
                device_index_);
            ring_fences_.resize(ring_size_);
            {
                VkFenceCreateInfo fCI{};
                fCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
                fCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
                for (uint32_t i = 0; i < ring_size_; ++i) {
                    vkCreateFence(ctx.getDevice(), &fCI, nullptr, &ring_fences_[i]);
                }
            }

            ring_iterations_.assign(ring_size_, 0);
            ring_output_sizes_.assign(ring_size_, 0);

            // Allocate descriptor pool and sets for each buffer pair
            {
                VkDescriptorPoolSize ps{};
                ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                ps.descriptorCount = pipeline->totalDescriptorCount * ring_size_;

                VkDescriptorPoolCreateInfo poolCI{};
                poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                poolCI.poolSizeCount = 1;
                poolCI.pPoolSizes = &ps;
                poolCI.maxSets = ring_size_;

                vkCreateDescriptorPool(ctx.getDevice(), &poolCI, nullptr, &ring_descriptor_pool_);

                std::vector<VkDescriptorSetLayout> layouts(ring_size_,
                                                           pipeline->descriptorSetLayout);
                VkDescriptorSetAllocateInfo allocInfo{};
                allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                allocInfo.descriptorPool = ring_descriptor_pool_;
                allocInfo.descriptorSetCount = ring_size_;
                allocInfo.pSetLayouts = layouts.data();

                ring_descriptor_sets_.resize(ring_size_);
                vkAllocateDescriptorSets(ctx.getDevice(), &allocInfo, ring_descriptor_sets_.data());

                for (uint32_t i = 0; i < ring_size_; ++i) {
                    auto& pair = (*buffer_ring_)[i];
                    VkDescriptorBufferInfo outDbi{};
                    outDbi.buffer = pair.output->getBuffer();
                    outDbi.offset = 0;
                    outDbi.range = pair.output->getSize();

                    std::vector<VkWriteDescriptorSet> writes;

                    VkWriteDescriptorSet wdsOut{};
                    wdsOut.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wdsOut.dstSet = ring_descriptor_sets_[i];
                    wdsOut.dstBinding = 0;
                    wdsOut.descriptorCount = 1;
                    wdsOut.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    wdsOut.pBufferInfo = &outDbi;
                    writes.push_back(wdsOut);

                    VkDescriptorBufferInfo inDbi{};
                    inDbi.buffer = pair.input->getBuffer();
                    inDbi.offset = 0;
                    inDbi.range = pair.input->getSize();

                    for (uint32_t b = 1; b < pipeline->bindings.size(); ++b) {
                        VkWriteDescriptorSet wdsIn{};
                        wdsIn.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        wdsIn.dstSet = ring_descriptor_sets_[i];
                        wdsIn.dstBinding = b;
                        wdsIn.descriptorCount = 1;
                        wdsIn.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        wdsIn.pBufferInfo = &inDbi;
                        writes.push_back(wdsIn);
                    }

                    vkUpdateDescriptorSets(ctx.getDevice(), static_cast<uint32_t>(writes.size()),
                                           writes.data(), 0, nullptr);
                }
            }

            // --------------------------------------------------
            // Reuse or create a single "max-sized" input staging
            // and output staging buffer for all iterations.
            // This drastically reduces creation/destroy overhead.
            // If chunkSize changes, we might recreate them,
            // or do something more advanced. For simplicity, fix it now.
            // --------------------------------------------------
            size_t size_of_input = stagingBuffer.size();

            VkDeviceSize maxOutputBytes =
                static_cast<VkDeviceSize>(iterations * func->output_variant_size_bytes_);

            endOfStream = false;
            bool noMoreInput = false;

            // Now we loop: gather data in batches, copy into staging, dispatch, and
            // read results from earlier submissions when their fences signal.
            while (true) {
                auto& pair = (*buffer_ring_)[current_pair_];
                VkFence fence = ring_fences_[current_pair_];
                vkWaitForFences(ctx.getDevice(), 1, &fence, VK_TRUE, UINT64_MAX);

                // If this entry has pending results, read them now
                if (ring_iterations_[current_pair_] > 0) {
                    std::vector<uint8_t> outData(ring_output_sizes_[current_pair_], 0);
                    void* mapped = nullptr;
                    vkMapMemory(ctx.getDevice(), pair.output->getMemory(), 0,
                                ring_output_sizes_[current_pair_], 0, &mapped);
                    std::memcpy(outData.data(), mapped, ring_output_sizes_[current_pair_]);
                    vkUnmapMemory(ctx.getDevice(), pair.output->getMemory());
                    for (size_t i = 0; i < ring_iterations_[current_pair_]; i++) {
                        std::vector<uint8_t> outItem(
                            outData.begin() + i * func->output_variant_size_bytes_,
                            outData.begin() + (i + 1) * func->output_variant_size_bytes_);
                        std::vector<float> variant = outItem;
                        output_buffer_->push(std::make_shared<std::vector<float>>(std::move(variant)));
                    }
                    ring_iterations_[current_pair_] = 0;
                }

                vkResetFences(ctx.getDevice(), 1, &fence);

                if (noMoreInput) {
                    bool pending = false;
                    for (auto v : ring_iterations_) {
                        if (v > 0) {
                            pending = true;
                            break;
                        }
                    }
                    if (!pending) {
                        output_buffer_->push(nullptr);
                        break;
                    }
                    current_pair_ = (current_pair_ + 1) % ring_size_;
                    continue;
                }

                // Copy all param data contiguously into the input buffer
                {
                    void* mapped = nullptr;
                    vkMapMemory(ctx.getDevice(), pair.input->getMemory(), 0,
                                static_cast<VkDeviceSize>(size_of_input), 0, &mapped);
                    uint8_t* mappedChar = reinterpret_cast<uint8_t*>(mapped);
                    VkDeviceSize vk_size = static_cast<VkDeviceSize>(stagingBuffer.size());
                    std::memcpy(mappedChar, stagingBuffer.data(), vk_size);
                    vkUnmapMemory(ctx.getDevice(), pair.input->getMemory());
                }

                VkDescriptorSet descriptorSet = ring_descriptor_sets_[current_pair_];

                // Set the first push constant value to the iteration count (N)
                push_constants_[0] = static_cast<uint32_t>(iterations);

                runComputeDispatch(*pipeline, descriptorSet, fence);
                ring_iterations_[current_pair_] = iterations;
                ring_output_sizes_[current_pair_] = maxOutputBytes;

                // Build next chunk while GPU processes this one
                size_t next_iterations = 0;
                size_t next_record_size = 0;
                size_t next_offset = 0;
                stagingBuffer.clear();
                while (stagingBuffer.size() + record_size < buffer_size) {
                    if (endOfStream)
                        break;
                    for (size_t i = 0; i < paramCount; i++) {
                        if (endOfStream)
                            break;
                        std::vector<std::shared_ptr<std::vector<float>>> item =
                            input_buffers_[i]->getForGPU(1);
                        next_iterations++;
                        for (size_t j = 0; j < item.size(); j++) {
                            if (!item[j]) {
                                endOfStream = true;
                                break;
                            }
                            size_t one_record_size = getVariantSizeInBytes(*item[j]);
                            std::vector<uint8_t> tmp(one_record_size);
                            copyVariantToMemory(*item[j], tmp.data());
                            stagingBuffer.resize(stagingBuffer.size() + one_record_size);
                            std::memcpy(stagingBuffer.data() + next_offset, tmp.data(),
                                        one_record_size);
                            next_offset += one_record_size;
                            next_record_size += one_record_size;
                        }
                    }
                }

                if (endOfStream)
                    noMoreInput = true;

                iterations = next_iterations;
                record_size = next_record_size;
                size_of_input = stagingBuffer.size();
                maxOutputBytes =
                    static_cast<VkDeviceSize>(iterations * func->output_variant_size_bytes_);

                current_pair_ = (current_pair_ + 1) % ring_size_;
                buffer_ring_->next();
            }

            // cleanup

            for (auto f : ring_fences_) {
                if (f != VK_NULL_HANDLE) {
                    vkDestroyFence(ctx.getDevice(), f, nullptr);
                }
            }
            if (ring_descriptor_pool_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(ctx.getDevice(), ring_descriptor_pool_, nullptr);
                ring_descriptor_pool_ = VK_NULL_HANDLE;
            }

            done_ = true;
            cleanupWrapper();
        } catch (...) {
            cleanupWrapper();
            output_buffer_->push(nullptr);
            done_ = true;
        }
    }

    /**
     * @brief Run the compute job on the GPU.
     */
    void runComputeDispatch(const ComputePipeline& pipeline, VkDescriptorSet descriptorSet,
                            VkFence fence) {
        auto& ctx = VulkanContext::getInstance(device_index_);

        // Allocate cmd buffer
        VkCommandBufferAllocateInfo cmdAI{};
        cmdAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAI.commandPool = ctx.getCommandPool();
        cmdAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAI.commandBufferCount = 1;

        VkCommandBuffer cmdBuf;
        vkAllocateCommandBuffers(ctx.getDevice(), &cmdAI, &cmdBuf);

        // Record
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmdBuf, &beginInfo);

        if (push_constants_.size() > 0) {
            vkCmdPushConstants(cmdBuf, pipeline.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               static_cast<uint32_t>(push_constants_.size() * sizeof(uint32_t)),
                               push_constants_.data());
        }

        vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
        vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipelineLayout, 0,
                                1, &descriptorSet, 0, nullptr);

        vkCmdDispatch(cmdBuf, gx_, gy_, gz_);

        vkEndCommandBuffer(cmdBuf);

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmdBuf;

        vkQueueSubmit(ctx.getComputeQueue(), 1, &si, fence);
        vkFreeCommandBuffers(ctx.getDevice(), ctx.getCommandPool(), 1, &cmdBuf);
    }

    /**
     * @brief Run the compute job on the CPU.
     */
    void runCPUFallback(const GPUFunction& func, size_t paramCount) {
        while (true) {
            std::vector<std::vector<float>> paramVec;
            paramVec.reserve(paramCount);

            // Gather paramCount items
            for (size_t i = 0; i < paramCount; i++) {
                auto item = input_buffers_[i]->getForGPU(1);
                if (!item[0]) {
                    // got nullptr => end of stream
                    output_buffer_->push(nullptr);
                    return;
                }
                paramVec.push_back(std::move(*item[i]));
            }

            // CPU fallback
            std::vector<float> result = func.cpuFallback(paramVec);

            // push result
            output_buffer_->push(std::make_shared<std::vector<float>>(std::move(result)));
        }
    }

    /**
     * @brief Destroy Vulkan resources allocated by this job.
     */
    void cleanupVulkanResources() {
        auto& ctx = VulkanContext::getInstance(device_index_);
        for (auto f : ring_fences_) {
            if (f != VK_NULL_HANDLE) {
                vkDestroyFence(ctx.getDevice(), f, nullptr);
            }
        }
        ring_fences_.clear();
        if (ring_descriptor_pool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(ctx.getDevice(), ring_descriptor_pool_, nullptr);
            ring_descriptor_pool_ = VK_NULL_HANDLE;
        }
    }

    float mem_percent_;
    std::vector<uint32_t> push_constants_;

    std::string kernelName_;
    std::shared_ptr<GPUBuffer> output_buffer_;
    std::vector<std::shared_ptr<GPUBuffer>> input_buffers_;
    uint32_t gx_, gy_, gz_;

    uint32_t ring_size_ = 1;
    std::unique_ptr<ShaderBufferRing> buffer_ring_;
    std::vector<VkFence> ring_fences_;
    VkDescriptorPool ring_descriptor_pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> ring_descriptor_sets_;
    std::vector<size_t> ring_iterations_;
    std::vector<VkDeviceSize> ring_output_sizes_;
    size_t current_pair_ = 0;

    std::thread worker_;
    std::atomic<bool> done_{false};
    std::atomic<bool> started_{false};
    uint32_t device_index_ = 0;
};

} // namespace psyne