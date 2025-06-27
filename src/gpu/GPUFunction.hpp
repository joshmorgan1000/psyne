#pragma once

#include "GPUBuffer.hpp"
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace psyne {

/**
 * @class GPUFunction
 *
 * @brief Represents a GPU function that can be executed on the GPU, with a CPU fallback.
 * These must be compiled shaders (see the shaders folder), which output a Shaders.hpp file,
 * but the metadata (such as parameters and CPU fallback functions) must be defined in this
 * file.
 *
 * @author jmorgan
 */
class GPUFunction {
  public:
    using CPUFallbackFunc = std::function<std::vector<float>(const std::vector<std::vector<float>>&)>;

    GPUFunction(std::vector<std::vector<float>> params, size_t push_constants,
                const std::vector<uint8_t>& shaderBinary, uint32_t groupCountX,
                uint32_t groupCountY, uint32_t groupCountZ, CPUFallbackFunc cpuImpl,
                size_t output_variant_size_bytes)
        : parameters(std::move(params)), push_constants_(push_constants), shader(shaderBinary),
          groupCountX_(groupCountX), groupCountY_(groupCountY), groupCountZ_(groupCountZ),
          cpuFallback(std::move(cpuImpl)), output_variant_size_bytes_(output_variant_size_bytes) {}

    std::vector<std::vector<float>> parameters;
    size_t push_constants_;
    std::vector<uint8_t> shader;
    CPUFallbackFunc cpuFallback;
    size_t output_variant_size_bytes_;
    uint32_t groupCountX_ = 1;
    uint32_t groupCountY_ = 1;
    uint32_t groupCountZ_ = 1;
};

/**
 * @class GPUFunctionRegistry
 *
 * @brief A singleton registry of GPU functions that can be executed on the GPU, with CPU fallbacks.
 */
class GPUFunctionRegistry {
  public:
    /**
     * @brief Get the singleton instance of the GPUFunctionRegistry.
     *
     * @return GPUFunctionRegistry&
     */
    static GPUFunctionRegistry& getInstance() {
        static GPUFunctionRegistry instance;
        return instance;
    }

    /**
     * @brief Get the GPUFunction with the given name.
     *
     * @param name  The name of the GPUFunction to retrieve.
     * @return const GPUFunction*
     */
    const GPUFunction* get(const std::string& name) const {
        auto it = registry_.find(name);
        if (it == registry_.end()) {
            return nullptr;
        }
        return &it->second;
    }

    /**
     * @brief Register a GPUFunction with the given name.
     *
     * @param name  The name of the GPUFunction to register.
     * @param func  The GPUFunction to register.
     */
    void registerFunction(const std::string& name, GPUFunction func) {
        registry_.emplace(name, std::move(func));
    }

    /**
     * @brief Get all registered GPUFunctions.
     *
     * @return const std::unordered_map<std::string, GPUFunction>&
     */
    const std::unordered_map<std::string, GPUFunction>& all() const { return registry_; }

  private:
    std::unordered_map<std::string, GPUFunction> registry_;
    GPUFunctionRegistry() = default;
};

} // namespace psyne