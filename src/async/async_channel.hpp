#pragma once

#include <boost/asio/awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>

namespace psyne {
namespace async {

/**
 * @brief Async message handler configuration
 */
struct AsyncHandlerConfig {
    /// Maximum number of concurrent handlers (0 = unlimited)
    size_t max_concurrent_handlers = 0;

    /// Handler timeout (0 = no timeout)
    std::chrono::milliseconds handler_timeout{0};

    /// Use thread pool for handlers
    bool use_thread_pool = true;

    /// Thread pool to use (nullptr = use global pool)
    class PsynePool *thread_pool = nullptr;
};

/**
 * @brief Base class for async channel operations
 *
 * This mixin adds boost.asio coroutine support to channels
 */
template <typename ChannelType>
class AsyncChannelMixin {
public:
    using executor_type = boost::asio::io_context::executor_type;

    /**
     * @brief Construct with optional shared io_context
     * @param io_ctx Shared io_context (creates own if nullptr)
     */
    explicit AsyncChannelMixin(
        std::shared_ptr<boost::asio::io_context> io_ctx = nullptr)
        : io_context_(io_ctx ? io_ctx
                             : std::make_shared<boost::asio::io_context>()),
          owns_io_context_(!io_ctx) {}

    /**
     * @brief Get the executor for this channel
     */
    executor_type get_executor() const {
        return io_context_->get_executor();
    }

    /**
     * @brief Asynchronously receive a message (coroutine)
     * @tparam MessageType Type of message to receive
     * @param timeout Timeout duration (0 = no timeout)
     * @return Awaitable that yields optional message
     */
    template <typename MessageType>
    boost::asio::awaitable<std::optional<MessageType>> async_receive(
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
        auto *channel = static_cast<ChannelType *>(this);

        // If no timeout, try once and return
        if (timeout == std::chrono::milliseconds::zero()) {
            auto msg = channel->template receive<MessageType>(timeout);
            co_return msg;
        }

        // With timeout, poll periodically
        boost::asio::steady_timer timer(*io_context_);
        auto deadline = std::chrono::steady_clock::now() + timeout;

        while (std::chrono::steady_clock::now() < deadline) {
            // Try to receive
            auto msg = channel->template receive<MessageType>(
                std::chrono::milliseconds::zero());
            if (msg) {
                co_return msg;
            }

            // Wait a bit before retrying
            timer.expires_after(std::chrono::milliseconds(10));
            co_await timer.async_wait(boost::asio::use_awaitable);
        }

        co_return std::nullopt;
    }

    /**
     * @brief Asynchronously receive a single message (blocking coroutine)
     * @tparam MessageType Type of message to receive
     * @param timeout Timeout duration
     * @return Awaitable that yields optional message
     */
    template <typename MessageType>
    boost::asio::awaitable<std::optional<MessageType>> async_receive_single(
        std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
        return async_receive<MessageType>(timeout);
    }

    /**
     * @brief Register an async message handler with optional thread pool
     * @tparam MessageType Type of message to handle
     * @param handler Callback function
     * @param config Handler configuration
     * @return Token to stop the handler
     */
    template <typename MessageType>
    std::shared_ptr<bool>
    register_async_handler(std::function<void(MessageType &&)> handler,
                           const AsyncHandlerConfig &config = {}) {
        auto *channel = static_cast<ChannelType *>(this);
        auto running = std::make_shared<bool>(true);

        // Spawn coroutine for message handling
        boost::asio::co_spawn(
            *io_context_,
            handle_messages<MessageType>(channel, handler, running, config),
            boost::asio::detached);

        return running;
    }

    /**
     * @brief Start processing async operations
     *
     * Call this if you're not running io_context elsewhere
     */
    void start_async() {
        if (owns_io_context_ && !io_thread_) {
            io_thread_ =
                std::make_unique<std::thread>([this]() { io_context_->run(); });
        }
    }

    /**
     * @brief Stop async operations
     */
    void stop_async() {
        io_context_->stop();
        if (io_thread_ && io_thread_->joinable()) {
            io_thread_->join();
            io_thread_.reset();
        }
    }

protected:
    std::shared_ptr<boost::asio::io_context> io_context_;
    bool owns_io_context_;
    std::unique_ptr<std::thread> io_thread_;

private:
    /**
     * @brief Coroutine to handle incoming messages
     */
    template <typename MessageType>
    boost::asio::awaitable<void> handle_messages(
        ChannelType *channel, std::function<void(MessageType &&)> handler,
        std::shared_ptr<bool> running, const AsyncHandlerConfig &config) {
        size_t active_handlers = 0;

        while (*running && !channel->is_stopped()) {
            // Receive message
            auto msg = co_await async_receive<MessageType>(
                std::chrono::milliseconds(100));

            if (msg) {
                // Check concurrent handler limit
                if (config.max_concurrent_handlers > 0 &&
                    active_handlers >= config.max_concurrent_handlers) {
                    // Skip this message or queue it
                    continue;
                }

                active_handlers++;

                // Process in thread pool or inline
                if (config.use_thread_pool) {
                    auto pool = config.thread_pool;
                    if (!pool) {
                        // Use global thread pool from pthread.hpp
                        extern PsynePool &get_global_thread_pool();
                        pool = &get_global_thread_pool();
                    }

                    pool->enqueue([handler, msg = std::move(*msg),
                                   &active_handlers]() mutable {
                        handler(std::move(msg));
                        active_handlers--;
                    });
                } else {
                    // Process inline
                    handler(std::move(*msg));
                    active_handlers--;
                }
            }
        }
    }
};

/**
 * @brief Helper to add async support to existing channel types
 */
template <typename BaseChannel>
class AsyncChannel : public BaseChannel,
                     public AsyncChannelMixin<AsyncChannel<BaseChannel>> {
public:
    using BaseChannel::BaseChannel;

    explicit AsyncChannel(
        const std::string &uri, size_t buffer_size,
        std::shared_ptr<boost::asio::io_context> io_ctx = nullptr)
        : BaseChannel(uri, buffer_size),
          AsyncChannelMixin<AsyncChannel<BaseChannel>>(io_ctx) {}
};

} // namespace async
} // namespace psyne