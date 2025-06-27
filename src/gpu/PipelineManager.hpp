#pragma once

#include "GlobalFunctionRegistry.hpp"
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

namespace psyne {

/**
 * @brief A struct that defines the layout of a descriptor binding.
 */
struct BindingDefinition {
    uint32_t bindingIndex;
    VkDescriptorType descriptorType;
    uint32_t descriptorCount;
    VkShaderStageFlags stageFlags;
};

/**
 * @brief Data for one compute pipeline.
 *
 * A compute pipeline is a collection of Vulkan objects that define a compute shader. It
 * is used to dispatch compute work on the GPU, and associates a shader module with a
 * descriptor set layout, pipeline layout, and descriptor pool.
 */
struct ComputePipeline {
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

    std::vector<BindingDefinition> bindings;
    uint32_t totalDescriptorCount = 0;
};

/**
 * @class PipelineManager
 *
 * @brief A manager for compute pipelines. Helps create, store, and destroy pipelines.
 *
 * @author jmorgan
 */
class PipelineManager {
  public:
    /**
     * @brief Constructor. Requires a Vulkan device.
     */
    PipelineManager(VkDevice device) : device_(device) {}

    /**
     * @brief Destructor. Destroys all pipelines.
     */
    ~PipelineManager() { destroyAll(); }

    PipelineManager(const PipelineManager&) = delete;
    PipelineManager& operator=(const PipelineManager&) = delete;

    /**
     * @brief Get or create a compute pipeline.
     */
    ComputePipeline* getOrCreate(const std::string& name, const std::vector<uint8_t>& userSpirv,
                                 const std::vector<BindingDefinition>& userBindings,
                                 const size_t& pushConstantsSize) {
        std::lock_guard<std::mutex> lock(mutex_);

        // 1) Check if we already have a pipeline for `name`
        auto it = pipelines_.find(name);
        if (it != pipelines_.end()) {
            return &it->second;
        }

        // 2) If the user provided an explicit SPIR-V, use it. Otherwise, fallback to
        // GPUFunctionRegistry
        std::vector<uint8_t> finalSpirv;
        const auto* func = GPUFunctionRegistry::getInstance().get(name);
        if (!userSpirv.empty()) {
            finalSpirv = userSpirv;
        } else {
            // fallback to registry
            if (!func) {
                throw std::runtime_error(
                    "PipelineManager: No SPIR-V provided and no GPUFunction found for name: " +
                    name);
            }
            if (func->shader.empty()) {
                throw std::runtime_error(
                    "PipelineManager: GPUFunction->shader is empty for name: " + name);
            }
            finalSpirv = func->shader;
        }

        if (finalSpirv.empty()) {
            throw std::runtime_error("PipelineManager: SPIR-V is empty for " + name);
        }

        // 3) Build a set of binding definitions.
        //    If userBindings is non-empty, we trust it. Otherwise, create a fallback
        std::vector<BindingDefinition> finalBindings;
        if (!userBindings.empty()) {
            // user gave explicit bindings
            finalBindings = userBindings;
        } else {
            // "Smart" fallback approach:
            // e.g., if we have a GPUFunction with N parameters, each one might get a binding:
            //  - binding=0 => output
            //  - binding=i+1 => input param i
            if (!func) {
                throw std::runtime_error(
                    "PipelineManager: No binding defs + no GPUFunction => can't guess bindings.");
            }

            // First, define an output binding at index 0
            BindingDefinition outBD;
            outBD.bindingIndex = 0;
            outBD.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            outBD.descriptorCount = 1;
            outBD.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            finalBindings.push_back(outBD);

            // Then, for each param in the GPUFunction, create an input binding
            for (size_t i = 0; i < func->parameters.size(); i++) {
                BindingDefinition inBD;
                inBD.bindingIndex = static_cast<uint32_t>(i + 1);
                inBD.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                inBD.descriptorCount = 1;
                inBD.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                finalBindings.push_back(inBD);
            }
        }

        // 4) Create shader module
        VkShaderModuleCreateInfo modCI{};
        modCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        modCI.codeSize = finalSpirv.size();
        modCI.pCode = reinterpret_cast<const uint32_t*>(finalSpirv.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device_, &modCI, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("PipelineManager: Failed to create shader module for " + name);
        }

        // 5) Create descriptor set layout
        std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
        layoutBindings.reserve(finalBindings.size());
        uint32_t totalDescCount = 0;

        for (const auto& bd : finalBindings) {
            VkDescriptorSetLayoutBinding lb{};
            lb.binding = bd.bindingIndex;
            lb.descriptorType = bd.descriptorType;
            lb.descriptorCount = bd.descriptorCount;
            lb.stageFlags = bd.stageFlags;
            layoutBindings.push_back(lb);

            totalDescCount += bd.descriptorCount;
        }

        VkDescriptorSetLayoutCreateInfo layoutCI{};
        layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutCI.bindingCount = static_cast<uint32_t>(layoutBindings.size());
        layoutCI.pBindings = layoutBindings.data();

        VkDescriptorSetLayout descLayout;
        if (vkCreateDescriptorSetLayout(device_, &layoutCI, nullptr, &descLayout) != VK_SUCCESS) {
            vkDestroyShaderModule(device_, shaderModule, nullptr);
            throw std::runtime_error(
                "PipelineManager: Failed to create descriptor set layout for " + name);
        }

        // 5.5) Set push constants
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = static_cast<uint32_t>(pushConstantsSize);

        // 6) Create pipeline layout
        VkPipelineLayoutCreateInfo plc{};
        plc.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plc.setLayoutCount = 1;
        plc.pSetLayouts = &descLayout;
        plc.pushConstantRangeCount = 1;
        plc.pPushConstantRanges = &pushConstantRange;

        VkPipelineLayout pipelineLayout;
        if (vkCreatePipelineLayout(device_, &plc, nullptr, &pipelineLayout) != VK_SUCCESS) {
            vkDestroyDescriptorSetLayout(device_, descLayout, nullptr);
            vkDestroyShaderModule(device_, shaderModule, nullptr);
            throw std::runtime_error("PipelineManager: Failed to create pipeline layout for " +
                                     name);
        }

        // 7) Create compute pipeline
        VkPipelineShaderStageCreateInfo stageCI{};
        stageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageCI.module = shaderModule;
        stageCI.pName = "main"; // entrypoint

        VkComputePipelineCreateInfo cpCI{};
        cpCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpCI.stage = stageCI;
        cpCI.layout = pipelineLayout;

        VkPipeline pipeline;
        if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpCI, nullptr, &pipeline) !=
            VK_SUCCESS) {
            vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
            vkDestroyDescriptorSetLayout(device_, descLayout, nullptr);
            vkDestroyShaderModule(device_, shaderModule, nullptr);
            throw std::runtime_error("PipelineManager: Failed to create compute pipeline for " +
                                     name);
        }

        // 8) Create descriptor pool
        // If your finalBindings include various descriptor types, you need to count them.
        // For simplicity, assume all are STORAGE_BUFFER.
        // We allocate enough descriptors for totalDescCount.
        std::vector<VkDescriptorPoolSize> poolSizes;
        {
            // Count how many STORAGE_BUFFER we need
            size_t storageCount = 0;
            for (const auto& bd : finalBindings) {
                if (bd.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                    storageCount += bd.descriptorCount;
                }
                // if you might have other descriptor types, accumulate them too
            }
            if (storageCount > 0) {
                VkDescriptorPoolSize ps{};
                ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                ps.descriptorCount = static_cast<uint32_t>(storageCount);
                poolSizes.push_back(ps);
            }
            // similarly for UNIFORM_BUFFER, SAMPLED_IMAGE, etc. if needed
        }

        VkDescriptorPoolCreateInfo poolCI{};
        poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolCI.pPoolSizes = poolSizes.data();
        poolCI.maxSets = 1; // One descriptor set

        VkDescriptorPool descriptorPool;
        if (vkCreateDescriptorPool(device_, &poolCI, nullptr, &descriptorPool) != VK_SUCCESS) {
            // cleanup
            vkDestroyPipeline(device_, pipeline, nullptr);
            vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
            vkDestroyDescriptorSetLayout(device_, descLayout, nullptr);
            vkDestroyShaderModule(device_, shaderModule, nullptr);
            throw std::runtime_error("PipelineManager: Failed to create descriptor pool for " +
                                     name);
        }

        // 9) Allocate descriptor set
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descLayout;

        VkDescriptorSet descriptorSet;
        if (vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSet) != VK_SUCCESS) {
            vkDestroyDescriptorPool(device_, descriptorPool, nullptr);
            vkDestroyPipeline(device_, pipeline, nullptr);
            vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
            vkDestroyDescriptorSetLayout(device_, descLayout, nullptr);
            vkDestroyShaderModule(device_, shaderModule, nullptr);
            throw std::runtime_error("PipelineManager: Failed to allocate descriptor set for " +
                                     name);
        }

        // 10) Build & store pipeline
        ComputePipeline cp;
        cp.shaderModule = shaderModule;
        cp.descriptorSetLayout = descLayout;
        cp.pipelineLayout = pipelineLayout;
        cp.pipeline = pipeline;
        cp.descriptorPool = descriptorPool;
        cp.descriptorSet = descriptorSet;
        cp.bindings = finalBindings;
        cp.totalDescriptorCount = totalDescCount;

        auto [insIt, _] = pipelines_.emplace(name, cp);
        return &insIt->second;
    }

    /**
     * @brief Finds a pipeline by name.
     */
    ComputePipeline* find(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pipelines_.find(name);
        if (it == pipelines_.end()) {
            return nullptr;
        }
        return &it->second;
    }

    /**
     * @brief Destroys a pipeline by name.
     */
    void destroy(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pipelines_.find(name);
        if (it == pipelines_.end()) {
            return;
        }
        teardown(it->second);
        pipelines_.erase(it);
    }

    /**
     * @brief Destroys all pipelines.
     */
    void destroyAll() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& kv : pipelines_) {
            teardown(kv.second);
        }
        pipelines_.clear();
    }

  private:
    VkDevice device_;
    std::mutex mutex_;
    std::unordered_map<std::string, ComputePipeline> pipelines_;

    /**
     * @brief Cleans up a ComputePipeline object.
     */
    void teardown(ComputePipeline& cp) {
        if (cp.descriptorPool) {
            vkDestroyDescriptorPool(device_, cp.descriptorPool, nullptr);
        }
        if (cp.pipeline) {
            vkDestroyPipeline(device_, cp.pipeline, nullptr);
        }
        if (cp.pipelineLayout) {
            vkDestroyPipelineLayout(device_, cp.pipelineLayout, nullptr);
        }
        if (cp.descriptorSetLayout) {
            vkDestroyDescriptorSetLayout(device_, cp.descriptorSetLayout, nullptr);
        }
        if (cp.shaderModule) {
            vkDestroyShaderModule(device_, cp.shaderModule, nullptr);
        }
        cp = {}; // reset
    }
};

} // namespace psyne