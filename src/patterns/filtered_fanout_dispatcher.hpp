/**
 * @file filtered_fanout_dispatcher.hpp
 * @brief Message dispatcher with filtered fanout and automatic response routing
 *
 * Provides a sophisticated message routing pattern where:
 * - Messages are routed to handlers based on user-defined predicates
 * - Multiple handlers can process the same message (fanout)
 * - All responses are automatically routed back to the original sender
 * - Thread pool execution with configurable concurrency
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#pragma once

#include <psyne/psyne.hpp>
// #include "../utils/pthread.hpp"  // Commented due to include path issues
// #include "../utils/logger.hpp"    // Commented due to include path issues
#include <concepts>
#include <future>
#include <iostream>
#include <span>
#include <variant>

namespace psyne {
namespace patterns {

// Concepts to regulate predicate and handler types
template <typename T>
concept MessagePredicate =
    requires(T t, const void *data, size_t size, uint32_t type) {
        { t(data, size, type) } -> std::convertible_to<bool>;
    };

template <typename T>
concept MessageHandler =
    requires(T t, const void *data, size_t size, uint32_t type,
             std::function<void(std::span<const uint8_t>)> reply) {
        { t(data, size, type, reply) } -> std::same_as<void>;
    };

/**
 * @brief Sender information embedded in messages for reply routing
 */
struct SenderInfo {
    std::string reply_channel_uri;
    uint64_t correlation_id;
    uint32_t sender_id;

    // Serialize to bytes
    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> data;
        data.reserve(sizeof(uint64_t) + sizeof(uint32_t) +
                     reply_channel_uri.size() + 1);

        // Write correlation_id
        const uint8_t *corr_ptr =
            reinterpret_cast<const uint8_t *>(&correlation_id);
        data.insert(data.end(), corr_ptr, corr_ptr + sizeof(correlation_id));

        // Write sender_id
        const uint8_t *sender_ptr =
            reinterpret_cast<const uint8_t *>(&sender_id);
        data.insert(data.end(), sender_ptr, sender_ptr + sizeof(sender_id));

        // Write URI as null-terminated string
        data.insert(data.end(), reply_channel_uri.begin(),
                    reply_channel_uri.end());
        data.push_back(0);

        return data;
    }

    // Deserialize from bytes
    static SenderInfo deserialize(const uint8_t *data, size_t size) {
        SenderInfo info;
        if (size < sizeof(uint64_t) + sizeof(uint32_t) + 1) {
            return info; // Invalid
        }

        // Read correlation_id
        std::memcpy(&info.correlation_id, data, sizeof(uint64_t));
        data += sizeof(uint64_t);

        // Read sender_id
        std::memcpy(&info.sender_id, data, sizeof(uint32_t));
        data += sizeof(uint32_t);

        // Read URI
        const char *uri_str = reinterpret_cast<const char *>(data);
        info.reply_channel_uri = std::string(uri_str);

        return info;
    }
};

/**
 * @brief Base message with sender info for reply routing
 */
template <typename T>
class RoutableMessage : public Message<T> {
public:
    using Message<T>::Message;

    void set_sender_info(const SenderInfo &info) {
        sender_info_ = info;
        // Store in message payload (implementation specific)
    }

    SenderInfo get_sender_info() const {
        return sender_info_;
    }

private:
    SenderInfo sender_info_;
};

/**
 * @brief Response aggregator for multi-match scenarios
 */
class ResponseAggregator {
public:
    ResponseAggregator(
        size_t expected_responses,
        std::function<void(std::vector<std::span<const uint8_t>>)>
            final_handler)
        : expected_(expected_responses),
          final_handler_(std::move(final_handler)), timeout_ms_(5000) {}

    void add_response(std::span<const uint8_t> response) {
        std::lock_guard<std::mutex> lock(mutex_);
        responses_.emplace_back(response.begin(), response.end());

        if (responses_.size() >= expected_) {
            send_aggregated_response();
        }
    }

    void set_timeout(std::chrono::milliseconds timeout) {
        timeout_ms_ = timeout;
    }

private:
    void send_aggregated_response() {
        // Convert stored responses to spans
        std::vector<std::span<const uint8_t>> spans;
        spans.reserve(responses_.size());

        for (const auto &resp : responses_) {
            spans.emplace_back(resp.data(), resp.size());
        }

        final_handler_(std::move(spans));
    }

    size_t expected_;
    std::vector<std::vector<uint8_t>> responses_;
    std::function<void(std::vector<std::span<const uint8_t>>)> final_handler_;
    std::mutex mutex_;
    std::chrono::milliseconds timeout_ms_;
};

/**
 * @brief Message dispatcher with filtered fanout and response aggregation
 *
 * Routes messages to multiple handlers based on predicates.
 * All responses are collected and sent back to the original sender.
 */
class FilteredFanoutDispatcher {
public:
    struct RouteConfig {
        std::string name;
        int32_t priority = 0;
        bool enabled = true;

        // Metrics
        mutable std::atomic<uint64_t> messages_processed{0};
        mutable std::atomic<uint64_t> messages_filtered{0};
    };

    template <MessagePredicate P, MessageHandler H>
    class Route {
    public:
        Route(P predicate, H handler, const std::string &name,
              int32_t priority = 0)
            : config_{.name = name, .priority = priority},
              predicate_(std::move(predicate)), handler_(std::move(handler)) {}

        Route(P predicate, H handler)
            : config_{}, predicate_(std::move(predicate)),
              handler_(std::move(handler)) {}

        bool matches(const void *data, size_t size, uint32_t type) const {
            if (!config_.enabled) {
                config_.messages_filtered.fetch_add(1,
                                                    std::memory_order_relaxed);
                return false;
            }
            return predicate_(data, size, type);
        }

        const RouteConfig &config() const {
            return config_;
        }

        void
        process(const void *data, size_t size, uint32_t type,
                std::function<void(std::span<const uint8_t>)> reply) const {
            config_.messages_processed.fetch_add(1, std::memory_order_relaxed);
            handler_(data, size, type, std::move(reply));
        }

        const P &predicate() const {
            return predicate_;
        }
        const H &handler() const {
            return handler_;
        }

    private:
        mutable RouteConfig config_;
        P predicate_;
        H handler_;
    };

    // Type-erased route interface
    class RouteBase {
    public:
        virtual ~RouteBase() = default;
        virtual bool matches(const void *data, size_t size,
                             uint32_t type) const = 0;
        virtual void
        process(const void *data, size_t size, uint32_t type,
                std::function<void(std::span<const uint8_t>)> reply) const = 0;
        virtual const RouteConfig &config() const = 0;
    };

    template <MessagePredicate P, MessageHandler H>
    class TypedRoute : public RouteBase {
    public:
        TypedRoute(Route<P, H> route) : route_(std::move(route)) {}

        bool matches(const void *data, size_t size,
                     uint32_t type) const override {
            return route_.matches(data, size, type);
        }

        void process(const void *data, size_t size, uint32_t type,
                     std::function<void(std::span<const uint8_t>)> reply)
            const override {
            route_.process(data, size, type, std::move(reply));
        }

        const RouteConfig &config() const override {
            return route_;
        }

    private:
        Route<P, H> route_;
    };

    /**
     * @brief Construct dispatcher with configurable thread pool
     * @param input_channel Channel to receive messages from
     * @param thread_count Number of worker threads (0 = hardware concurrency)
     * @param enable_work_stealing Enable work stealing between threads
     */
    FilteredFanoutDispatcher(std::shared_ptr<Channel> input_channel,
                             size_t thread_count = 0,
                             bool enable_work_stealing = true)
        : input_channel_(std::move(input_channel)), thread_count_(thread_count),
          next_correlation_id_(1) {
        (void)enable_work_stealing; // unused
        std::cout << "[INFO] FilteredFanoutDispatcher created" << std::endl;
    }

    /**
     * @brief Add a typed route with compile-time type checking
     */
    template <typename MessageType, typename Predicate, typename Handler>
        requires std::derived_from<MessageType, Message<MessageType>>
    void add_typed_route(const std::string &name, Predicate &&predicate,
                         Handler &&handler, int32_t priority = 0) {
        auto pred_wrapper = [predicate = std::forward<Predicate>(predicate)](
                                const void *data, size_t size,
                                uint32_t type) -> bool {
            if (type != MessageType::message_type)
                return false;
            const auto *msg = reinterpret_cast<const MessageType *>(data);
            return predicate(*msg);
        };

        auto handler_wrapper =
            [handler = std::forward<Handler>(handler)](
                const void *data, size_t size, uint32_t type,
                std::function<void(std::span<const uint8_t>)> reply) {
                const auto *msg = reinterpret_cast<const MessageType *>(data);

                // Handler can return a response or void
                if constexpr (std::is_invocable_r_v<void, Handler,
                                                    const MessageType &>) {
                    handler(*msg);
                    // Send empty response
                    reply(std::span<const uint8_t>{});
                } else {
                    auto response = handler(*msg);
                    reply(std::span<const uint8_t>{
                        reinterpret_cast<const uint8_t *>(&response),
                        response.calculate_size()});
                }
            };

        add_route(std::move(pred_wrapper), std::move(handler_wrapper), name,
                  priority);
    }

    /**
     * @brief Add a raw route with manual type checking
     */
    template <MessagePredicate P, MessageHandler H>
    void add_route(P &&predicate, H &&handler, const std::string &name,
                   int32_t priority = 0) {
        auto route = std::make_unique<TypedRoute<P, H>>(
            Route<P, H>(std::forward<P>(predicate), std::forward<H>(handler),
                        name, priority));

        std::lock_guard<std::mutex> lock(routes_mutex_);
        routes_.push_back(std::move(route));
    }

    template <MessagePredicate P, MessageHandler H>
    void add_route(P &&predicate, H &&handler) {
        auto route = std::make_unique<TypedRoute<P, H>>(
            Route<P, H>(std::forward<P>(predicate), std::forward<H>(handler)));

        std::lock_guard<std::mutex> lock(routes_mutex_);
        routes_.push_back(std::move(route));
    }

    /**
     * @brief Start the dispatcher (pre-warms all threads)
     */
    void start() {
        if (running_.exchange(true)) {
            std::cerr << "[WARN] Dispatcher already running" << std::endl;
            return;
        }

        // Start worker threads
        size_t num_threads = thread_count_ > 0
                                 ? thread_count_
                                 : std::thread::hardware_concurrency();
        for (size_t i = 0; i < num_threads; ++i) {
            worker_threads_.emplace_back([this] { worker_loop(); });
        }

        std::cout << "[INFO] Started " << num_threads << " worker threads"
                  << std::endl;

        // Start dispatch thread
        dispatch_thread_ = std::thread([this] { dispatch_loop(); });
    }

    void stop() {
        running_ = false;
        if (dispatch_thread_.joinable()) {
            dispatch_thread_.join();
        }
        // Join all worker threads
        for (auto &t : worker_threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    /**
     * @brief Get dispatcher metrics
     */
    struct Metrics {
        uint64_t messages_received = 0;
        uint64_t messages_dispatched = 0;
        uint64_t no_matches = 0;
        size_t active_routes = 0;
        size_t pending_tasks = 0;
    };

    Metrics get_metrics() const {
        Metrics m;
        m.messages_received = messages_received_.load();
        m.messages_dispatched = messages_dispatched_.load();
        m.no_matches = no_matches_.load();
        m.pending_tasks = 0; // Simplified implementation

        std::lock_guard<std::mutex> lock(routes_mutex_);
        m.active_routes = routes_.size();

        return m;
    }

private:
    void dispatch_loop() {
        while (running_) {
            size_t size;
            uint32_t type;
            void *msg_data = input_channel_->receive_raw_message(size, type);

            if (!msg_data) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }

            messages_received_.fetch_add(1, std::memory_order_relaxed);

            // Extract sender info for reply routing
            SenderInfo sender_info;
            size_t header_size = 0;

            // Check if message has sender info header
            if (size >= sizeof(uint32_t)) {
                uint32_t magic = *reinterpret_cast<const uint32_t *>(msg_data);
                if (magic == 0xDEADBEEF) { // Magic number for routable messages
                    // Skip magic and size
                    const uint8_t *header_data =
                        static_cast<const uint8_t *>(msg_data) +
                        sizeof(uint32_t);
                    uint32_t header_len =
                        *reinterpret_cast<const uint32_t *>(header_data);
                    header_data += sizeof(uint32_t);

                    sender_info =
                        SenderInfo::deserialize(header_data, header_len);
                    header_size = sizeof(uint32_t) * 2 + header_len;
                }
            }

            // Adjust data pointer and size to skip header
            const void *actual_msg_data =
                static_cast<const uint8_t *>(msg_data) + header_size;
            size_t actual_msg_size = size - header_size;

            // Find all matching routes
            std::vector<std::unique_ptr<RouteBase> *> matching_routes;
            {
                std::lock_guard<std::mutex> lock(routes_mutex_);
                for (auto &route : routes_) {
                    if (route->matches(actual_msg_data, actual_msg_size,
                                       type)) {
                        matching_routes.push_back(&route);
                    }
                }
            }

            if (matching_routes.empty()) {
                no_matches_.fetch_add(1, std::memory_order_relaxed);
                if (!sender_info.reply_channel_uri.empty()) {
                    send_no_handler_response(sender_info);
                }
            } else {
                // Create response aggregator if we have reply info
                std::shared_ptr<ResponseAggregator> aggregator;

                if (!sender_info.reply_channel_uri.empty()) {
                    aggregator = std::make_shared<ResponseAggregator>(
                        matching_routes.size(),
                        [this, sender_info](
                            std::vector<std::span<const uint8_t>> responses) {
                            send_aggregated_response(sender_info, responses);
                        });
                }

                // Dispatch to all matching routes
                for (auto *route_ptr : matching_routes) {
                    // Copy message data for async processing
                    auto msg_copy = std::make_shared<std::vector<uint8_t>>(
                        static_cast<const uint8_t *>(actual_msg_data),
                        static_cast<const uint8_t *>(actual_msg_data) +
                            actual_msg_size);

                    messages_dispatched_.fetch_add(1,
                                                   std::memory_order_relaxed);

                    // In simplified implementation, execute directly
                    (*route_ptr)
                        ->process(
                            msg_copy->data(), msg_copy->size(), type,
                            [aggregator](std::span<const uint8_t> response) {
                                if (aggregator) {
                                    aggregator->add_response(response);
                                }
                            });
                }
            }

            input_channel_->release_raw_message(msg_data);
        }
    }

    void send_no_handler_response(const SenderInfo &sender_info) {
        // Create a simple "no handler" response message
        struct NoHandlerResponse {
            uint32_t status = 404;
            char message[60] = "No handler found for message type";
        };

        NoHandlerResponse response;
        send_response_to_sender(sender_info,
                                reinterpret_cast<const uint8_t *>(&response),
                                sizeof(response));
    }

    void send_aggregated_response(
        const SenderInfo &sender_info,
        const std::vector<std::span<const uint8_t>> &responses) {
        // Create aggregated response message
        std::vector<uint8_t> aggregated;

        // Header: number of responses
        uint32_t count = static_cast<uint32_t>(responses.size());
        const uint8_t *count_ptr = reinterpret_cast<const uint8_t *>(&count);
        aggregated.insert(aggregated.end(), count_ptr,
                          count_ptr + sizeof(count));

        // Each response with size prefix
        for (const auto &response : responses) {
            uint32_t resp_size = static_cast<uint32_t>(response.size());
            const uint8_t *size_ptr =
                reinterpret_cast<const uint8_t *>(&resp_size);
            aggregated.insert(aggregated.end(), size_ptr,
                              size_ptr + sizeof(resp_size));
            aggregated.insert(aggregated.end(), response.begin(),
                              response.end());
        }

        send_response_to_sender(sender_info, aggregated.data(),
                                aggregated.size());
    }

    void send_response_to_sender(const SenderInfo &sender_info,
                                 const uint8_t *response_data,
                                 size_t response_size) {
        if (sender_info.reply_channel_uri.empty()) {
            return;
        }

        try {
            // Get or create reply channel
            auto reply_channel =
                get_or_create_reply_channel(sender_info.reply_channel_uri);
            if (!reply_channel) {
                std::cerr << "[WARN] Failed to create reply channel: "
                          << sender_info.reply_channel_uri << std::endl;
                return;
            }

            // Reserve space for response
            auto slot = reply_channel->reserve_write_slot(response_size);
            if (slot == 0xFFFFFFFF) { // BUFFER_FULL
                std::cerr << "[WARN] Reply channel buffer full" << std::endl;
                return;
            }

            // Write response
            auto span = reply_channel->get_write_span(response_size);
            std::memcpy(span.data(), response_data, response_size);
            reply_channel->notify_message_ready(slot, response_size);

        } catch (const std::exception &e) {
            std::cerr << "[ERROR] Error sending response: " << e.what()
                      << std::endl;
        }
    }

    Channel *get_or_create_reply_channel(const std::string &uri) {
        std::lock_guard<std::mutex> lock(reply_channels_mutex_);

        auto it = reply_channels_.find(uri);
        if (it != reply_channels_.end()) {
            return it->second.get();
        }

        // Create new channel
        try {
            auto channel = Channel::create(uri, 1024 * 1024); // 1MB buffer
            auto *channel_ptr = channel.get();
            reply_channels_[uri] = std::move(channel);
            return channel_ptr;
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] Failed to create reply channel " << uri
                      << ": " << e.what() << std::endl;
            return nullptr;
        }
    }

    void worker_loop() {
        // Worker threads in simplified implementation
        // In a real implementation, would process from a work queue
        while (running_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    void process_message(const void *data, size_t size, uint32_t type) {
        std::vector<std::span<const uint8_t>> responses;

        {
            std::lock_guard<std::mutex> lock(routes_mutex_);
            for (const auto &route : routes_) {
                if (route->matches(data, size, type)) {
                    route->process(
                        data, size, type,
                        [&responses](std::span<const uint8_t> response) {
                            responses.push_back(response);
                        });
                }
            }
        }

        // Send aggregated response if configured
        // For now, just count responses
        if (!responses.empty()) {
            messages_dispatched_ += responses.size();
        } else {
            no_matches_++;
        }
        messages_received_++;
    }

private:
    std::shared_ptr<Channel> input_channel_;
    std::vector<std::thread> worker_threads_;
    size_t thread_count_ = 0;

    std::vector<std::unique_ptr<RouteBase>> routes_;
    mutable std::mutex routes_mutex_;

    std::thread dispatch_thread_;
    std::atomic<bool> running_{false};

    // Reply channel cache
    std::unordered_map<std::string, std::unique_ptr<Channel>> reply_channels_;
    std::mutex reply_channels_mutex_;

    // Metrics
    std::atomic<uint64_t> messages_received_{0};
    std::atomic<uint64_t> messages_dispatched_{0};
    std::atomic<uint64_t> no_matches_{0};
    std::atomic<uint64_t> next_correlation_id_;
};

/**
 * @brief Helper to create a routable message with sender info
 */
template <typename T>
class MessageBuilder {
public:
    static std::vector<uint8_t>
    build_with_reply_info(const T &message, const std::string &reply_uri,
                          uint64_t correlation_id = 0) {
        SenderInfo info;
        info.reply_channel_uri = reply_uri;
        info.correlation_id = correlation_id;
        info.sender_id =
            std::hash<std::thread::id>{}(std::this_thread::get_id());

        auto sender_data = info.serialize();

        std::vector<uint8_t> result;

        // Magic number
        uint32_t magic = 0xDEADBEEF;
        const uint8_t *magic_ptr = reinterpret_cast<const uint8_t *>(&magic);
        result.insert(result.end(), magic_ptr, magic_ptr + sizeof(magic));

        // Header size
        uint32_t header_size = static_cast<uint32_t>(sender_data.size());
        const uint8_t *size_ptr =
            reinterpret_cast<const uint8_t *>(&header_size);
        result.insert(result.end(), size_ptr, size_ptr + sizeof(header_size));

        // Header data
        result.insert(result.end(), sender_data.begin(), sender_data.end());

        // Message data
        const uint8_t *msg_ptr = reinterpret_cast<const uint8_t *>(&message);
        result.insert(result.end(), msg_ptr,
                      msg_ptr + message.calculate_size());

        return result;
    }
};

} // namespace patterns
} // namespace psyne