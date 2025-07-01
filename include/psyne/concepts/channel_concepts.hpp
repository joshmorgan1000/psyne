#pragma once

#include <concepts>
#include <type_traits>
#include <functional>
#include <vector>
#include <boost/asio.hpp>
#include <chrono>

namespace psyne::concepts {

/**
 * @brief Concept for message types that can be used in channels
 */
template<typename T, typename S>
concept MessageType = requires {
    // Must have a size known at compile time
    sizeof(T);
    
    // Must be movable for RAII
    std::is_move_constructible_v<T>;
    
    // Must be destructible
    std::is_destructible_v<T>;
} && requires(S& substrate) {
    // Must be constructible with substrate
    T(substrate);
} && requires(S& substrate, int arg1, float arg2) {
    // Must support variadic construction with substrate + args
    T(substrate, arg1, arg2);
};

/**
 * @brief Simpler message concept for basic types
 */
template<typename T>
concept BasicMessageType = requires {
    sizeof(T);
    std::is_move_constructible_v<T>;
    std::is_destructible_v<T>;
    
    // Can be default constructed
    T{};
};

/**
 * @brief Concept for channel substrates
 */
template<typename S, typename T>
concept SubstrateType = requires(S substrate, T* msg_ptr, size_t size_bytes, 
                                std::vector<std::function<void(T*)>>& listeners) {
    // Memory management
    { substrate.allocate_slab(size_bytes) } -> std::convertible_to<T*>;
    { substrate.deallocate_slab(msg_ptr) } -> std::same_as<void>;
    
    // Message sending (sync and async)
    { substrate.send_message(msg_ptr, listeners) } -> std::same_as<void>;
    { substrate.async_send_message(msg_ptr, listeners) } -> std::same_as<boost::asio::awaitable<void>>;
    
    // Substrate capabilities
    { substrate.needs_serialization() } -> std::same_as<bool>;
    { substrate.is_zero_copy() } -> std::same_as<bool>;
    { substrate.is_cross_process() } -> std::same_as<bool>;
    { substrate.name() } -> std::convertible_to<const char*>;
    
    // Optional lifecycle methods
    { substrate.initialize() } -> std::same_as<void>;
    { substrate.shutdown() } -> std::same_as<void>;
};

/**
 * @brief Concept for channel patterns
 */
template<typename P, typename T, typename S>
concept PatternType = requires(P pattern, T* slab, size_t max_messages, 
                              boost::asio::io_context& io_context,
                              std::chrono::milliseconds timeout) {
    // Core pattern operations
    { pattern.try_allocate(slab, max_messages) } -> std::convertible_to<T*>;
    { pattern.try_receive() } -> std::convertible_to<T*>;
    
    // Async operations
    { pattern.async_receive(io_context, timeout) } -> std::same_as<boost::asio::awaitable<T*>>;
    { pattern.receive_blocking(timeout) } -> std::convertible_to<T*>;
    
    // Pattern capabilities
    { pattern.needs_locks() } -> std::same_as<bool>;
    { pattern.max_producers() } -> std::same_as<size_t>;
    { pattern.max_consumers() } -> std::same_as<size_t>;
    { pattern.name() } -> std::convertible_to<const char*>;
    
    // Pattern state
    { pattern.size() } -> std::same_as<size_t>;
    { pattern.empty() } -> std::same_as<bool>;
    { pattern.full() } -> std::same_as<bool>;
    
    // Producer/Consumer management (optional)
    { pattern.register_producer() } -> std::same_as<size_t>;
    { pattern.register_consumer() } -> std::same_as<size_t>;
    { pattern.unregister_producer(size_t{}) } -> std::same_as<void>;
    { pattern.unregister_consumer(size_t{}) } -> std::same_as<void>;
};

/**
 * @brief Concept for complete channel configuration
 */
template<typename T, typename S, typename P>
concept ChannelConfiguration = (MessageType<T, S> || BasicMessageType<T>) && 
                              SubstrateType<S, T> && 
                              PatternType<P, T, S>;

/**
 * @brief Concept for plugin substrates (user-defined)
 */
template<typename S, typename T>
concept PluginSubstrate = SubstrateType<S, T> && requires {
    // Plugin substrates should have a version identifier
    S::plugin_version;
    
    // And a unique name
    S::plugin_name;
};

/**
 * @brief Concept for plugin patterns (user-defined)
 */
template<typename P, typename T, typename S>
concept PluginPattern = PatternType<P, T, S> && requires {
    // Plugin patterns should have a version identifier
    P::plugin_version;
    
    // And a unique name
    P::plugin_name;
};

/**
 * @brief Concept for high-performance substrates
 */
template<typename S, typename T>
concept HighPerformanceSubstrate = SubstrateType<S, T> && requires(S substrate) {
    // High-performance substrates should be zero-copy
    { substrate.is_zero_copy() } -> std::same_as<bool>;
    
    // And provide performance metrics
    { substrate.get_bandwidth_mbps() } -> std::convertible_to<double>;
    { substrate.get_latency_ns() } -> std::convertible_to<uint64_t>;
} && requires {
    // Should be zero-copy
    std::same_as<decltype(std::declval<S>().is_zero_copy()), bool>;
};

/**
 * @brief Concept for real-time patterns
 */
template<typename P, typename T, typename S>
concept RealTimePattern = PatternType<P, T, S> && requires(P pattern) {
    // Real-time patterns should provide latency guarantees
    { pattern.max_latency_ns() } -> std::convertible_to<uint64_t>;
    { pattern.is_lock_free() } -> std::same_as<bool>;
} && requires {
    // Should be lock-free for real-time
    !std::same_as<decltype(std::declval<P>().needs_locks()), std::true_type>;
};

} // namespace psyne::concepts