#pragma once

#include "GPUBuffer.hpp"
#include "GPUComputeJob.hpp"
#include "GPUFunction.hpp"
#include "VulkanContext.hpp"
#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <variant>
#include <vulkan/vulkan.h>
#include <memory>
#include <optional>
#include <vector>

namespace psyne {

/**
 * @class Wrapper
 *
 * @brief Probably the easiest way to run a Vulkan GPU compute job.
 *
 * Usage:
 *
 * 1) Write your shader and place it in the shaders folder. This project will need to be rebuilt in
 * order for it to be included in this library. Otherwise you will have to call
 *          GPUFunctionRegistry::getInstance().registerFunction() yourself. There are conventions
 * you must follow when writing the shader.
 *
 * 2) Create a Wrapper object, specifying the name of your shader.
 *
 * 3) Add producer functions either via the constructor or the addProducer() method. The order in
 * which these producers are added must match the order in which they are declared in the shader.
 * The first producer (binding 0) is reserved for the output buffer.
 *
 * 4) Call the getResult() method to run the job and get the result. This will return an optional
 * containing the result, or an empty optional when the stream is done (job is complete).
 *
 * @author jmorgan
 */
class Wrapper {
  public:
    /**
     * @brief You can construct with just a name, but you must add producers before attempting to
     * get any values from the job.
     */
    Wrapper(const std::string& shader_name, uint32_t device_index = 0)
        : shader_name_(shader_name), mem_percent_(0.1), ring_size_(2),
          device_index_(device_index),
          output_buffer_(std::make_shared<GPUBuffer>([]() { return nullptr; }, true)) {}

    /**
     * @brief Construct with a name and a list of producers.
     *
     * @param shader_name   The name of the shader to run.
     * @param producers     A list of producer functions that generate std::vector<float> objects.
     * @param mem_percent   The percentage of the GPU memory to use for this job.
     */
    Wrapper(const std::string& shader_name, std::vector<std::function<std::vector<float>()>> producers,
            std::vector<uint32_t> push_constants = {}, const float& mem_percent = 0.1f,
            uint32_t ring_size = 2, uint32_t device_index = 0)
        : shader_name_(shader_name), producers_(producers.begin(), producers.end()),
          pushConstants_(push_constants), mem_percent_(mem_percent), ring_size_(ring_size),
          device_index_(device_index),
          output_buffer_(std::make_shared<GPUBuffer>([]() { return nullptr; }, true)) {}

    /**
     * @brief Add a producer function to the job. They must be added in the same order that they are
     *        declared in the shader, reserving 0 for the output buffer.
     */
    void addProducer(std::function<std::vector<float>()> producer) { producers_.emplace_back(std::move(producer)); }

    /**
     * @brief Sets the percentage of GPU memory this GPU job will try to use. Default is 0.1.
     *        Don't set this to 1.0, it will not work.
     */
    void setMemoryPercent(const float& mem_percent) { mem_percent_ = mem_percent; }

    /**
     * @brief Sets the push constants for the job. These are 32-bit values that can be used in the
     * shader as function parameters. The order of the values must match the order in which they are
     * declared in the shader.
     */
    void setPushConstants(const std::vector<uint32_t>& pushConstants) {
        pushConstants_ = pushConstants;
    }

    /**
     * @brief Sets the number of rotating buffer pairs to use.
     */
    void setRingSize(uint32_t ring_size) { ring_size_ = ring_size; }

    /**
     * @brief Run the job (if it isn't already running) and get the result.
     */
    std::optional<std::vector<float>> getResult() {
        if (!job_) {
            if (producers_.empty()) {
                throw std::runtime_error("You must have at least one producer to run the job.");
            }
            // Initialize the job
            std::vector<std::shared_ptr<GPUBuffer>> input_buffers;
            while (!producers_.empty()) {
                std::function<std::vector<float>()> producer = std::move(producers_.front());
                producers_.pop_front();
                input_buffers.emplace_back(std::make_shared<GPUBuffer>(std::move(producer), true));
            }
            job_ = std::make_unique<GPUComputeJob>(shader_name_, output_buffer_, input_buffers,
                                                   pushConstants_, mem_percent_, ring_size_,
                                                   device_index_);
            job_->start();
        }
        std::shared_ptr<std::vector<float>> result = output_buffer_->get().get();
        if (!result)
            return std::nullopt;
        return *result;
    }

  private:
    std::string shader_name_;
    std::deque<std::function<std::vector<float>()>> producers_;
    std::shared_ptr<GPUBuffer> output_buffer_;
    float mem_percent_;
    uint32_t ring_size_ = 2;
    std::unique_ptr<GPUComputeJob> job_;
    std::vector<uint32_t> pushConstants_;
    uint32_t device_index_ = 0;
};

} // namespace psyne