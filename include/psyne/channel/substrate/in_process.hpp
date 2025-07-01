#pragma once

#include "substrate_base.hpp"
#include "global/logger.hpp"
#include <cstdlib>

namespace psyne::substrate {

/**
 * @brief In-process memory substrate - fastest possible
 */
template<typename T>
class InProcess : public SubstrateBase<T> {
public:
    T* allocate_slab(size_t size_bytes) override {
        return static_cast<T*>(std::aligned_alloc(alignof(T), size_bytes));
    }
    
    void deallocate_slab(T* ptr) override {
        std::free(ptr);
    }
    
    void send_message(T* msg_ptr, std::vector<std::function<void(T*)>>& listeners) override {
        // Direct notification - zero latency
        for (auto& listener : listeners) {
            listener(msg_ptr);
        }
    }
    
    boost::asio::awaitable<void> async_send_message(T* msg_ptr, 
                                                   std::vector<std::function<void(T*)>>& listeners) override {
        // For in-process, async send is just sync send
        send_message(msg_ptr, listeners);
        co_return;
    }
    
    bool needs_serialization() const override { return false; }
    bool is_zero_copy() const override { return true; }
    bool is_cross_process() const override { return false; }
    const char* name() const override { return "InProcess"; }
};

} // namespace psyne::substrate