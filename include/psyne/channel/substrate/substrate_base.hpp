#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <boost/asio.hpp>

namespace psyne::substrate {

/**
 * @brief Base class for all channel substrates
 * 
 * Defines the interface that all substrates must implement.
 * All substrates can send messages and should support async receive.
 */
template<typename T>
class SubstrateBase {
public:
    virtual ~SubstrateBase() = default;

    /**
     * @brief Allocate memory slab for messages
     */
    virtual T* allocate_slab(size_t size_bytes) = 0;
    
    /**
     * @brief Deallocate memory slab
     */
    virtual void deallocate_slab(T* ptr) = 0;
    
    /**
     * @brief Send message to listeners (sync)
     */
    virtual void send_message(T* msg_ptr, std::vector<std::function<void(T*)>>& listeners) = 0;
    
    /**
     * @brief Send message async (all substrates should support this)
     */
    virtual boost::asio::awaitable<void> async_send_message(T* msg_ptr, 
                                                           std::vector<std::function<void(T*)>>& listeners) = 0;
    
    /**
     * @brief Initialize substrate (optional)
     */
    virtual void initialize() {}
    
    /**
     * @brief Shutdown substrate (optional)  
     */
    virtual void shutdown() {}
    
    /**
     * @brief Substrate capabilities
     */
    virtual bool needs_serialization() const = 0;
    virtual bool is_zero_copy() const = 0;
    virtual bool is_cross_process() const = 0;
    virtual const char* name() const = 0;
};

} // namespace psyne::substrate