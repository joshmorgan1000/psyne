#pragma once

/**
 * @file channel_v3.hpp
 * @brief Clean composition-based channel system
 * 
 * Channel<MessageType, Substrate, Pattern> where:
 * - Substrates inherit from SubstrateBase (organized by category)
 * - Patterns inherit from PatternBase (organized by category) 
 * - Messages are traits (organized by category)
 */

#include "substrate/substrate_base.hpp"
#include "substrate/in_process.hpp"
#include "substrate/tcp.hpp"
#include "pattern/pattern_base.hpp"
#include "pattern/spsc.hpp"
#include "message/numeric_types.hpp"
#include "concepts/channel_concepts.hpp"
#include "global/logger.hpp"

#include <memory>
#include <functional>
#include <boost/asio.hpp>

namespace psyne {

/**
 * @brief Unified channel with composition-based design
 * 
 * @tparam T Message type (any trivially copyable type)
 * @tparam S Substrate type (inherits from SubstrateBase)
 * @tparam P Pattern type (inherits from PatternBase)
 */
template<typename T, typename S, typename P>
    requires concepts::ChannelConfiguration<T, S, P>
class Channel {
public:

    using Substrate = S;
    using Pattern = P;

    /**
     * @brief Create channel with memory slab
     */
    explicit Channel(uint16_t size_factor = 0, size_t ring_size = 1024) 
        : size_factor_(size_factor) {
        
        // Allocate typed memory slab using substrate
        size_t slab_size = (size_factor + 1) * 32 * 1024 * 1024;
        slab_ = substrate_.allocate_slab(slab_size);
        if (!slab_) {
            throw std::bad_alloc();
        }
        
        max_messages_ = slab_size / sizeof(T);
        
        // Initialize pattern with ring size
        pattern_ = std::make_unique<P>(ring_size);
        
        LOG_INFO("Created Channel<{}, {}, {}> with {}MB slab ({} messages)", 
                 typeid(T).name(), substrate_.name(), pattern_->name(),
                 slab_size / (1024 * 1024), max_messages_);
    }

    ~Channel() {
        if (slab_) {
            substrate_.deallocate_slab(slab_);
        }
    }

    // Non-copyable
    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;

    /**
     * @brief Allocate message using pattern
     */
    T* allocate() {
        return pattern_->try_allocate(slab_, max_messages_);
    }

    /**
     * @brief Try to receive message (non-blocking)
     */
    T* try_receive() {
        return pattern_->try_receive();
    }

    /**
     * @brief Async receive with awaitable
     */
    boost::asio::awaitable<T*> async_receive(
        boost::asio::io_context& io_context,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
        return pattern_->async_receive(io_context, timeout);
    }

    /**
     * @brief Blocking receive with timeout
     */
    T* receive(std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
        return pattern_->receive_blocking(timeout);
    }

    /**
     * @brief Send message using substrate
     */
    void send(T* msg_ptr) {
        substrate_.send_message(msg_ptr, listeners_);
    }

    /**
     * @brief Async send message using substrate
     */
    boost::asio::awaitable<void> async_send(T* msg_ptr) {
        return substrate_.async_send_message(msg_ptr, listeners_);
    }

    /**
     * @brief Register message listener
     */
    void register_listener(std::function<void(T*)> listener) {
        listeners_.push_back(std::move(listener));
    }

    /**
     * @brief Producer/Consumer management (delegates to pattern)
     */
    size_t register_producer() {
        return pattern_->register_producer();
    }

    size_t register_consumer() {
        return pattern_->register_consumer();
    }

    /**
     * @brief Channel state
     */
    size_t size() const { return pattern_->size(); }
    bool empty() const { return pattern_->empty(); }
    bool full() const { return pattern_->full(); }

    /**
     * @brief Substrate and pattern info
     */
    bool needs_serialization() const { return substrate_.needs_serialization(); }
    bool is_zero_copy() const { return substrate_.is_zero_copy(); }
    bool is_cross_process() const { return substrate_.is_cross_process(); }
    bool needs_locks() const { return pattern_->needs_locks(); }

    size_t capacity() const { return max_messages_; }
    const char* substrate_name() const { return substrate_.name(); }
    const char* pattern_name() const { return pattern_->name(); }

private:
    uint16_t size_factor_;
    T* slab_ = nullptr;
    size_t max_messages_ = 0;
    
    S substrate_;
    std::unique_ptr<P> pattern_;
    std::vector<std::function<void(T*)>> listeners_;
};

/**
 * @brief Message wrapper for RAII and convenience
 */
template<typename T, typename S, typename P>
class Message {
public:
    explicit Message(Channel<T, S, P>& channel) : channel_(channel) {
        data_ = channel_.allocate();
        if (!data_) {
            throw std::runtime_error("Channel full - cannot allocate message");
        }
        new (data_) T{};
    }

    template<typename... Args>
    Message(Channel<T, S, P>& channel, Args&&... args) : channel_(channel) {
        data_ = channel_.allocate();
        if (!data_) {
            throw std::runtime_error("Channel full - cannot allocate message");
        }
        new (data_) T(std::forward<Args>(args)...);
    }

    // Move-only
    Message(const Message&) = delete;
    Message& operator=(const Message&) = delete;
    
    Message(Message&& other) noexcept 
        : channel_(other.channel_), data_(other.data_) {
        other.data_ = nullptr;
    }

    T* operator->() { return data_; }
    const T* operator->() const { return data_; }
    T& operator*() { return *data_; }
    const T& operator*() const { return *data_; }

    /**
     * @brief Send message - delegates to channel substrate
     */
    void send() {
        if (data_) {
            channel_.send(data_);
            data_ = nullptr;
        }
    }

    /**
     * @brief Async send message
     */
    boost::asio::awaitable<void> async_send() {
        if (data_) {
            co_await channel_.async_send(data_);
            data_ = nullptr;
        }
    }

    bool valid() const { return data_ != nullptr; }

private:
    Channel<T, S, P>& channel_;
    T* data_ = nullptr;
};

// Convenience type aliases using the new structure
namespace substrate = psyne::substrate;
namespace pattern = psyne::pattern;
namespace message = psyne::message;

// Common channel types
template<typename T> 
using FastChannel = Channel<T, substrate::InProcess<T>, pattern::SPSC<T, substrate::InProcess<T>>>;

template<typename T>
using NetworkChannel = Channel<T, substrate::TCP<T>, pattern::SPSC<T, substrate::TCP<T>>>;

// Factory functions
template<typename T>
std::shared_ptr<FastChannel<T>> make_fast_channel(uint16_t size_factor = 0) {
    return std::make_shared<FastChannel<T>>(size_factor);
}

template<typename T>
std::shared_ptr<NetworkChannel<T>> make_network_channel(uint16_t size_factor = 0) {
    return std::make_shared<NetworkChannel<T>>(size_factor);
}

} // namespace psyne