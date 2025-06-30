#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <psyne/psyne.hpp>
#include <regex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace psyne {
namespace routing {

/**
 * @class MessageFilter
 * @brief Base class for message filtering
 */
class MessageFilter {
public:
    virtual ~MessageFilter() = default;

    /**
     * @brief Check if a message passes the filter
     * @param type Message type ID
     * @param data Message data pointer
     * @param size Message size
     * @return true if message passes filter, false otherwise
     */
    virtual bool matches(uint32_t type, const void *data,
                         size_t size) const = 0;
};

/**
 * @class TypeFilter
 * @brief Filter messages by type ID
 */
class TypeFilter : public MessageFilter {
public:
    explicit TypeFilter(uint32_t type) : type_(type) {}

    bool matches(uint32_t type, const void *data, size_t size) const override {
        (void)data;
        (void)size;
        return type == type_;
    }

private:
    uint32_t type_;
};

/**
 * @class RangeFilter
 * @brief Filter messages by type ID range
 */
class RangeFilter : public MessageFilter {
public:
    RangeFilter(uint32_t min_type, uint32_t max_type)
        : min_type_(min_type), max_type_(max_type) {}

    bool matches(uint32_t type, const void *data, size_t size) const override {
        (void)data;
        (void)size;
        return type >= min_type_ && type <= max_type_;
    }

private:
    uint32_t min_type_;
    uint32_t max_type_;
};

/**
 * @class SizeFilter
 * @brief Filter messages by size
 */
class SizeFilter : public MessageFilter {
public:
    SizeFilter(size_t min_size, size_t max_size)
        : min_size_(min_size), max_size_(max_size) {}

    bool matches(uint32_t type, const void *data, size_t size) const override {
        (void)type;
        (void)data;
        return size >= min_size_ && size <= max_size_;
    }

private:
    size_t min_size_;
    size_t max_size_;
};

/**
 * @class PredicateFilter
 * @brief Filter messages using custom predicate
 */
class PredicateFilter : public MessageFilter {
public:
    using Predicate = std::function<bool(uint32_t, const void *, size_t)>;

    explicit PredicateFilter(Predicate predicate) : predicate_(predicate) {}

    bool matches(uint32_t type, const void *data, size_t size) const override {
        return predicate_(type, data, size);
    }

private:
    Predicate predicate_;
};

/**
 * @class CompositeFilter
 * @brief Combine multiple filters with AND/OR logic
 */
class CompositeFilter : public MessageFilter {
public:
    enum class Mode { AND, OR };

    explicit CompositeFilter(Mode mode = Mode::AND) : mode_(mode) {}

    void add_filter(std::unique_ptr<MessageFilter> filter) {
        filters_.push_back(std::move(filter));
    }

    bool matches(uint32_t type, const void *data, size_t size) const override {
        if (filters_.empty())
            return true;

        if (mode_ == Mode::AND) {
            for (const auto &filter : filters_) {
                if (!filter->matches(type, data, size)) {
                    return false;
                }
            }
            return true;
        } else { // OR
            for (const auto &filter : filters_) {
                if (filter->matches(type, data, size)) {
                    return true;
                }
            }
            return false;
        }
    }

private:
    Mode mode_;
    std::vector<std::unique_ptr<MessageFilter>> filters_;
};

/**
 * @class MessageRoute
 * @brief Route configuration with filter and handler
 */
class MessageRoute {
public:
    using Handler = std::function<void(uint32_t type, void *data, size_t size)>;

    MessageRoute(std::unique_ptr<MessageFilter> filter, Handler handler)
        : filter_(std::move(filter)), handler_(handler) {}

    bool process(uint32_t type, void *data, size_t size) {
        if (filter_->matches(type, data, size)) {
            handler_(type, data, size);
            return true;
        }
        return false;
    }

private:
    std::unique_ptr<MessageFilter> filter_;
    Handler handler_;
};

/**
 * @class MessageRouter
 * @brief Routes messages from channels to appropriate handlers
 */
class MessageRouter {
public:
    MessageRouter() : running_(false) {}
    ~MessageRouter() {
        stop();
    }

    /**
     * @brief Add a route with type filter
     */
    template <typename MessageType>
    void add_route(std::function<void(MessageType &&)> handler) {
        auto filter = std::make_unique<TypeFilter>(MessageType::message_type);
        auto route_handler = [handler](uint32_t type, void *data, size_t size) {
            (void)type; // Type already filtered
            MessageType msg(data, size);
            handler(std::move(msg));
        };
        routes_.emplace_back(std::move(filter), route_handler);
    }

    /**
     * @brief Add a route with custom filter
     */
    void add_route(std::unique_ptr<MessageFilter> filter,
                   MessageRoute::Handler handler) {
        routes_.emplace_back(std::move(filter), handler);
    }

    /**
     * @brief Add a default route for unmatched messages
     */
    void set_default_route(MessageRoute::Handler handler) {
        default_handler_ = handler;
    }

    /**
     * @brief Start routing messages from a channel
     */
    void start(Channel &channel) {
        if (running_.exchange(true)) {
            return; // Already running
        }

        router_thread_ = std::thread([this, &channel]() {
            while (running_) {
                size_t size;
                uint32_t type;
                void *data = channel.receive_raw_message(size, type);

                if (!data) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }

                bool matched = false;
                for (auto &route : routes_) {
                    if (route.process(type, data, size)) {
                        matched = true;
                        break; // First match wins
                    }
                }

                if (!matched && default_handler_) {
                    default_handler_(type, data, size);
                }

                channel.release_raw_message(data);
            }
        });
    }

    /**
     * @brief Stop the router
     */
    void stop() {
        running_ = false;
        if (router_thread_.joinable()) {
            router_thread_.join();
        }
    }

    /**
     * @brief Get statistics
     */
    struct Stats {
        uint64_t messages_routed = 0;
        uint64_t messages_unmatched = 0;
    };

    Stats get_stats() const {
        return stats_;
    }

private:
    std::vector<MessageRoute> routes_;
    MessageRoute::Handler default_handler_;
    std::atomic<bool> running_;
    std::thread router_thread_;
    mutable Stats stats_;
};

/**
 * @class FilteredChannel
 * @brief Wrapper that adds filtering to any channel
 */
class FilteredChannel : public Channel {
public:
    FilteredChannel(std::unique_ptr<Channel> inner,
                    std::unique_ptr<MessageFilter> filter)
        : inner_(std::move(inner)), filter_(std::move(filter)) {}

    void stop() override {
        inner_->stop();
    }
    bool is_stopped() const override {
        return inner_->is_stopped();
    }
    const std::string &uri() const override {
        return inner_->uri();
    }
    ChannelType type() const override {
        return inner_->type();
    }
    ChannelMode mode() const override {
        return inner_->mode();
    }

    void *receive_raw_message(size_t &size, uint32_t &type) override {
        while (true) {
            void *data = inner_->receive_raw_message(size, type);
            if (!data)
                return nullptr;

            if (filter_->matches(type, data, size)) {
                return data;
            }

            // Message doesn't match filter, release and try again
            inner_->release_raw_message(data);
        }
    }

    void release_raw_message(void *handle) override {
        inner_->release_raw_message(handle);
    }

    bool has_metrics() const override {
        return inner_->has_metrics();
    }
    debug::ChannelMetrics get_metrics() const override {
        return inner_->get_metrics();
    }
    void reset_metrics() override {
        inner_->reset_metrics();
    }

protected:
    detail::ChannelImpl *impl() override {
        return inner_->get_impl();
    }
    const detail::ChannelImpl *impl() const override {
        return inner_->get_impl();
    }

private:
    std::unique_ptr<Channel> inner_;
    std::unique_ptr<MessageFilter> filter_;
};

/**
 * @brief Create a filtered channel
 */
inline std::unique_ptr<Channel>
create_filtered_channel(std::unique_ptr<Channel> channel,
                        std::unique_ptr<MessageFilter> filter) {
    return std::make_unique<FilteredChannel>(std::move(channel),
                                             std::move(filter));
}

} // namespace routing
} // namespace psyne