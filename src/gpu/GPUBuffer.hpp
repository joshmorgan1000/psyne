#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <variant>
#include <vector>
#include <vulkan/vulkan.h>

namespace psyne {

/**
 * @class GPUBuffer
 *
 * @brief A simple class that encapsulates a Vulkan buffer and device memory,
 * but also provides producer/consumer functions to move data between
 * CPU-side data structures and GPU memory.
 *
 * The idea is that the producer_ can generate CPU data (any of the above variant types),
 * and the consumer_ can read back from the GPU into CPU data structures,
 * if your application requires it.
 */
class GPUBuffer {
  public:
    /**
     * @brief Construct a GPUBuffer with the given producer and consumer functions.
     *
     * @param producer          A function that generates GPU data, std::shared_ptr<std::vector<float>>
     *                          values. It should return nullptr when it is done producing data.
     * @param is_stream         If false, this buffer is just a parameter buffer. It is not a stream
     *                          of data, but just a single buffer that is filled once and read once.
     * @param inputBufferLimit  The maximum number of headed-to-the-GPU data items to buffer.
     *                          If -1, no limit is enforced.
     * @param outputBufferLimit The maximum number of headed-to-the-CPU data items to buffer.
     *                          If -1, no limit is enforced.
     *
     * @note The constructor does not spawn any background thread. The producer
     *       is only invoked when `callProducer()` or other APIs request data.
     *       If you do not consume the data in the out_buffer_ and the
     *       `outputBufferLimit_` is reached, GPU shader operations will block
     *       until you consume the data.
     */
    GPUBuffer(std::function<std::vector<float>()> producer, bool is_stream, uint64_t inputBufferLimit = 4096,
              uint64_t outputBufferLimit = 4096)
        : producer_(producer != nullptr ? std::move(producer) : nullptr),
          inputBufferLimit_(inputBufferLimit), outputBufferLimit_(outputBufferLimit) {}

    // Copy constructor
    GPUBuffer(const GPUBuffer& other) {
        producer_ = other.producer_;
        inputBufferLimit_ = other.inputBufferLimit_;
        outputBufferLimit_ = other.outputBufferLimit_;
        input_eof_ = false;
        output_eof_ = false;
    }

    // Move constructor
    GPUBuffer(GPUBuffer&& other) noexcept
        : producer_(std::move(other.producer_)), inputBufferLimit_(other.inputBufferLimit_),
          outputBufferLimit_(other.outputBufferLimit_), input_eof_(false), output_eof_(false) {}

    /**
     * @brief Returns the next item in the output buffer.
     *
     * @return A future that resolves to the next item in the output buffer.
     */
    std::future<std::shared_ptr<std::vector<float>>> get() {
        auto fut = getMany(1);
        std::promise<std::shared_ptr<std::vector<float>>> single_promise;
        auto single_fut = single_promise.get_future();
        std::thread([fut = std::move(fut), single_promise = std::move(single_promise)]() mutable {
            auto vec = fut.get();
            if (vec.empty()) {
                single_promise.set_value(nullptr);
                return;
            }
            single_promise.set_value(std::move(vec.front()));
        }).detach();
        return single_fut;
    }

    /**
     * @brief Returns the next 'count' items in the output buffer.
     *
     * @param count  The number of items to return.
     *
     * @return A future that resolves to a vector of the next 'count' items in the output buffer.
     *
     * @note You want to be sure not to request more items than you expect to be in the buffer,
     *       otherwise this function will block until the buffer is filled, which will essentially
     *       means it could end up waiting forever.
     */
    std::future<std::vector<std::shared_ptr<std::vector<float>>>> getMany(uint64_t count) {
        std::unique_lock<std::mutex> lock(out_buffer_mutex_);
        if (out_buffer_.size() >= count) {
            std::vector<std::shared_ptr<std::vector<float>>> items;
            for (uint64_t i = 0; i < count; ++i) {
                items.push_back(out_buffer_.front());
                out_buffer_.pop();
            }
            std::promise<std::vector<std::shared_ptr<std::vector<float>>>> promise;
            auto fut = promise.get_future();
            promise.set_value(std::move(items));
            return fut;
        }
        if (output_eof_) {
            std::vector<std::shared_ptr<std::vector<float>>> items;
            while (!out_buffer_.empty() && items.size() < count) {
                items.push_back(out_buffer_.front());
                out_buffer_.pop();
            }
            std::promise<std::vector<std::shared_ptr<std::vector<float>>>> promise;
            auto fut = promise.get_future();
            promise.set_value(std::move(items));
            return fut;
        }
        PendingRequest pending;
        pending.count = count;
        auto fut = pending.promise.get_future();
        pending_requests_.push(std::move(pending));
        return fut;
    }

    /**
     * @brief Pushes an item into the output buffer. This should only be called by the
     * VulkanContext.
     *
     * @param item  The item to push into the output buffer.
     */
    void push(std::shared_ptr<std::vector<float>> data) {
        std::vector<std::pair<std::promise<std::vector<std::shared_ptr<std::vector<float>>>>,
                              std::vector<std::shared_ptr<std::vector<float>>>>>
            to_fulfill;
        std::unique_lock<std::mutex> lock(out_buffer_mutex_);
        if (!data) {
            output_eof_ = true;
            while (!pending_requests_.empty()) {
                auto& request = pending_requests_.front();
                std::vector<std::shared_ptr<std::vector<float>>> result;
                while (!out_buffer_.empty() && result.size() < request.count) {
                    result.push_back(out_buffer_.front());
                    out_buffer_.pop();
                }
                to_fulfill.emplace_back(std::move(request.promise), std::move(result));
                pending_requests_.pop();
            }
            return;
        }
        out_buffer_.push(std::move(data));
        while (!pending_requests_.empty()) {
            auto& request = pending_requests_.front();
            if (out_buffer_.size() >= request.count) {
                std::vector<std::shared_ptr<std::vector<float>>> collected;
                for (uint64_t i = 0; i < request.count; ++i) {
                    collected.push_back(out_buffer_.front());
                    out_buffer_.pop();
                }
                to_fulfill.emplace_back(std::move(request.promise), std::move(collected));
                pending_requests_.pop();
            } else {
                break;
            }
        }
        lock.unlock();
        for (auto& [promise, result] : to_fulfill) {
            promise.set_value(std::move(result));
        }
    }

    /**
     * @brief Pushes a vector of items into the output buffer. This should only be called by the
     * VulkanContext.
     *
     * @param items  The items to push into the output buffer.
     */
    void pushMany(std::vector<std::shared_ptr<std::vector<float>>> items) {
        std::unique_lock<std::mutex> lock(out_buffer_mutex_);
        for (auto& item : items) {
            if (!item) {
                output_eof_ = true;
                while (!pending_requests_.empty()) {
                    auto request = std::move(pending_requests_.front());
                    pending_requests_.pop();
                    std::vector<std::shared_ptr<std::vector<float>>> result;
                    while (!out_buffer_.empty() && result.size() < request.count) {
                        result.push_back(out_buffer_.front());
                        out_buffer_.pop();
                    }
                    request.promise.set_value(std::move(result));
                }
                return;
            }
            out_buffer_.push(std::move(item));
        }
        while (!pending_requests_.empty()) {
            auto& request = pending_requests_.front();
            if (out_buffer_.size() >= request.count) {
                std::vector<std::shared_ptr<std::vector<float>>> collected;
                for (uint64_t i = 0; i < request.count; ++i) {
                    collected.push_back(out_buffer_.front());
                    out_buffer_.pop();
                }
                auto promise = std::move(request.promise);
                pending_requests_.pop();
                lock.unlock();
                promise.set_value(std::move(collected));
                lock.lock();
            } else {
                break;
            }
        }
    }

    /**
     * @brief Returns actual values (not just pointers) from the input buffer for for the GPU, and
     * then calls fillInput() again to keep the buffer filled... unless the producer_ returns
     * nullptr.
     *
     * @param count  The number of items to return.
     *
     * @return A std::vector<std::vector<float>> of the next 'count' items in the input buffer, or the
     * next possible available until end of stream.
     */
    std::vector<std::shared_ptr<std::vector<float>>> getForGPU(uint64_t count) {
        std::vector<std::shared_ptr<std::vector<float>>> items;
        for (uint64_t i = 0; i < count; ++i) {
            callProducer();
            std::unique_lock<std::mutex> lock(in_buffer_mutex_);
            if (in_buffer_.empty()) {
                if (input_eof_) {
                    return items;
                }
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            items.push_back(in_buffer_.front());
            in_buffer_.pop();
        }
        return items;
    }

    /**
     * @brief Calls the producer function to fill the input buffer with one item.
     */
    void callProducer() {
        if (producer_) {
            std::vector<float> data = producer_();
            in_buffer_.push(std::make_shared<std::vector<float>>(std::move(data)));
        }
    }

    /**
     * @brief If false, this buffer is just a parameter buffer. It is not a stream of data, but just
     * a single buffer that is filled once and read once.
     */
    bool isStream() const { return is_stream_; }

  private:
    std::function<std::vector<float>()> producer_;
    bool is_stream_;
    uint64_t inputBufferLimit_;
    uint64_t outputBufferLimit_;
    std::queue<std::shared_ptr<std::vector<float>>> in_buffer_;
    std::mutex in_buffer_mutex_;
    std::queue<std::shared_ptr<std::vector<float>>> out_buffer_;
    std::mutex out_buffer_mutex_;
    std::atomic<bool> input_eof_{false};
    std::atomic<bool> output_eof_{false};
    struct PendingRequest {
        uint64_t count;
        std::promise<std::vector<std::shared_ptr<std::vector<float>>>> promise;
    };
    std::queue<PendingRequest> pending_requests_;
};

} // namespace psyne