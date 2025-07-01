#pragma once

#include <memory>
#include <boost/asio.hpp>
#include <chrono>

namespace psyne::pattern {

/**
 * @brief Base class for all channel patterns
 * 
 * Defines the interface that all patterns must implement.
 * All patterns support allocation, sync/async receive, and producer/consumer management.
 */
template<typename T, typename Substrate>
class PatternBase {
public:
    virtual ~PatternBase() = default;

    /**
     * @brief Try to allocate a message in the slab
     */
    virtual T* try_allocate(T* slab, size_t max_messages) = 0;
    
    /**
     * @brief Try to receive a message (non-blocking)
     */
    virtual T* try_receive() = 0;
    
    /**
     * @brief Async receive with awaitable (all patterns should support this)
     */
    virtual boost::asio::awaitable<T*> async_receive(
        boost::asio::io_context& io_context,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) = 0;
    
    /**
     * @brief Blocking receive with timeout
     */
    virtual T* receive_blocking(std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) = 0;
    
    /**
     * @brief Pattern capabilities
     */
    virtual bool needs_locks() const = 0;
    virtual size_t max_producers() const = 0;
    virtual size_t max_consumers() const = 0;
    virtual const char* name() const = 0;
    
    /**
     * @brief Pattern state
     */
    virtual size_t size() const = 0;
    virtual bool empty() const = 0;
    virtual bool full() const = 0;
    
    /**
     * @brief Producer/Consumer management (optional, pattern-specific)
     */
    virtual size_t register_producer() { return 0; }
    virtual size_t register_consumer() { return 0; }
    virtual void unregister_producer(size_t id) {}
    virtual void unregister_consumer(size_t id) {}
};

} // namespace psyne::pattern